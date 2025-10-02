"""
Created on Wed Oct  1 01:31:59 2025

@author: 周俊宇
"""



#模型函式庫
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time as ti
import numpy as np


from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms


#訓練資料載入器

class DeblurringDataset(Dataset):
    """
    一個用於圖像去模糊的自定義 PyTorch Dataset。
    它會讀取成對的模糊圖像和原始圖像。
    """
    def __init__(self, data_dir, transform=None):
        """
        初始化 DeblurringDataset。
        
        Args:
            data_dir (str): 包含子資料夾 'blurry' 和 'original' 的主資料夾路徑。
            transform (callable, optional): 應用於圖像的轉換。
        """
        self.data_dir = data_dir
        self.blurry_dir = os.path.join(data_dir, 'blurry')
        self.original_dir = os.path.join(data_dir, 'original')
        
        # 確保資料夾存在
        if not os.path.exists(self.blurry_dir) or not os.path.exists(self.original_dir):
            raise FileNotFoundError(f"找不到 'blurry' 或 'original' 資料夾。請確認路徑：{data_dir}")

        # 獲取模糊圖像的檔案列表
        # 我們只依賴這個列表來獲取圖像數量和檔名
        self.image_files = [f for f in os.listdir(self.blurry_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform
        
    def __len__(self):
        """返回資料集中的圖像對數量。"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        獲取單一的圖像對（模糊的輸入和原始的標籤）。
        """
        # 獲取模糊圖像的檔名
        blurry_file_name = self.image_files[idx]
        
        # 假設檔名格式是 '編號_blur.png'
        # 通過替換後綴來構造原始圖像的檔名
        if blurry_file_name.endswith('.png'):
            original_file_name = blurry_file_name.replace('_blur.png', '_orig.png')
        elif blurry_file_name.endswith('.jpg'):
            original_file_name = blurry_file_name.replace('_blur.jpg', '_orig.jpg')
        elif blurry_file_name.endswith('.jpeg'):
            original_file_name = blurry_file_name.replace('_blur.jpeg', '_orig.jpeg')
        else:
            # 如果後綴不是預期的，則拋出錯誤
            raise ValueError(f"不支援的檔名格式：{blurry_file_name}")

        blurry_image_path = os.path.join(self.blurry_dir, blurry_file_name)
        original_image_path = os.path.join(self.original_dir, original_file_name)
        
        # 檢查原始圖像是否存在，以確保資料集的完整性
        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"找不到對應的原始圖像：{original_image_path}")
        
        # 載入模糊圖像和原始圖像
        blurry_image = Image.open(blurry_image_path).convert('RGB')
        original_image = Image.open(original_image_path).convert('RGB')
        
        # 應用預處理轉換（例如：轉為張量）
        if self.transform:
            blurry_tensor = self.transform(blurry_image)
            original_tensor = self.transform(original_image)
        else:
            blurry_tensor = transforms.ToTensor()(blurry_image)
            original_tensor = transforms.ToTensor()(original_image)

        # 返回模糊圖像作為輸入，以及原始圖像作為標籤
        return blurry_tensor, original_tensor

# 修訂後的 SIDDenoisingDataset 類別

class SIDDenoisingDataset(Dataset):
    """
    用於SIDD降噪任務的自定義PyTorch Dataset。
    它會在初始化時預先載入所有檔案路徑，以提高效率。
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []

        # 在初始化時遍歷所有子資料夾，找到所有圖像路徑
        for scene_dir_name in os.listdir(data_dir):
            scene_dir_path = os.path.join(data_dir, scene_dir_name)
            if not os.path.isdir(scene_dir_path):
                continue
            
            noisy_image_path = None
            clean_image_path = None
            
            # 使用 os.walk 遞迴尋找檔案
            for root, _, files in os.walk(scene_dir_path):
                for file in files:
                    # 忽略副檔名大小寫，並檢查檔案名是否包含關鍵字
                    if 'NOISY_SRGB' in file.upper() and file.lower().endswith('.png'):
                        noisy_image_path = os.path.join(root, file)
                    elif 'GT_SRGB' in file.upper() and file.lower().endswith('.png'):
                        clean_image_path = os.path.join(root, file)
            
            if noisy_image_path and clean_image_path:
                self.image_paths.append((noisy_image_path, clean_image_path))
        
        print(f"SIDD資料集已載入 {len(self.image_paths)} 組圖像對。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        noisy_image_path, clean_image_path = self.image_paths[idx]
        
        noisy_image = Image.open(noisy_image_path).convert('RGB')
        clean_image = Image.open(clean_image_path).convert('RGB')

        if self.transform:
            noisy_tensor = self.transform(noisy_image)
            clean_tensor = self.transform(clean_image)
        else:
            noisy_tensor = transforms.ToTensor()(noisy_image)
            clean_tensor = transforms.ToTensor()(clean_image)

        return noisy_tensor, clean_tensor


#-----------------------------------------------------------
# 模型定義
# (此部分與你提供的程式碼相同，無需修改)
#-----------------------------------------------------------
class ConvResBlock(nn.Module):
    def __init__(self,channel):
        super(ConvResBlock, self).__init__()
        self.convlayer1=nn.Conv2d(channel,channel,kernel_size=11,padding=5)
        #self.norm1=nn.InstanceNorm2d(channel)
        self.convlayer2=nn.Conv2d(channel,channel, kernel_size=5,padding=2)
        #self.norm2=nn.InstanceNorm2d(channel)
        self.convlayer3=nn.Conv2d(channel,channel, kernel_size=5,padding=2)
        #self.norm3=nn.InstanceNorm2d(channel)
    def forward(self,x):
        output=self.convlayer1(x)
        output=F.leaky_relu((output))
        output=self.convlayer2(output)
        output=F.leaky_relu((output))
        output=self.convlayer3(output)
        output=output+x
        return output
        
class DenoiseBlock(nn.Module):
    def __init__(self,encode_channel,bottleneck_channel,decode_channel,encode_depth,bottleneck_depth,decode_depth):
        super(DenoiseBlock, self).__init__()
        self.adapterlayer1=nn.Conv2d(3,encode_channel,kernel_size=3,padding=1)
        self.EncodeBlock=nn.Sequential(*[ConvResBlock(encode_channel) for _ in range(encode_depth)])
        self.adapterlayer2=nn.Conv2d(encode_channel,bottleneck_channel,kernel_size=3,padding=1)
        self.BottleneckBlock=nn.Sequential(*[ConvResBlock(bottleneck_channel) for _ in range(bottleneck_depth)])
        self.adapterlayer3=nn.Conv2d(bottleneck_channel,decode_channel,kernel_size=3,padding=1)
        self.DecoserBlock=nn.Sequential(*[ConvResBlock(decode_channel) for _ in range(decode_depth)])
        self.RGBadapterlayer=nn.Conv2d(decode_channel,3,kernel_size=3,padding=1)
    def forward(self,x):
        residual1=x
        output=F.leaky_relu(self.adapterlayer1(x))
        output=self.EncodeBlock(output)
        residual2=output
        output=F.leaky_relu(self.adapterlayer2(output))
        output=self.BottleneckBlock(output)
        output=F.leaky_relu(self.adapterlayer3(output))
        output=self.DecoserBlock(output)+residual2
        output=self.RGBadapterlayer(output)
        return output+residual1

#-----------------------------------------------------------
# 訓練與驗證流程
# (此部分與之前提供給你的程式碼相同，無需修改)

#-----------------------------------------------------------
def main():
    start=ti.time()
    loss_curve=[]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用裝置: {device}')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 2. 載入兩個資料集
    # 文字去模糊數據集
    TextImage_DATA_DIR = r"C:\Users\周俊宇\Desktop\正則化降噪\文字去模糊數據集"
    text_train_dataset = DeblurringDataset(data_dir=os.path.join(TextImage_DATA_DIR, 'train'), transform=transform)
    text_val_dataset = DeblurringDataset(data_dir=os.path.join(TextImage_DATA_DIR, 'val'), transform=transform)

    
    BATCH_SIZE = 8
    # 4. 創建統一的DataLoader
    train_loader = DataLoader(text_train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=8)
    val_loader = DataLoader(text_val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=8)
    
    
    #資料載入檢查
    print(f"文字去模糊訓練集樣本數: {len(text_train_dataset)}")
    print(f"總訓練批次數: {len(train_loader)}")
    

    # 模型架構決定
    encode_channel = 12
    bottleneck_channel = 10
    decode_channel = 12
    encode_depth = 15
    bottleneck_depth = 5
    decode_depth = 15

    model = DenoiseBlock(encode_channel, bottleneck_channel, decode_channel, 
                         encode_depth, bottleneck_depth, decode_depth).to(device)

    MAE_criterion = nn.L1Loss()
    MSE_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    NUM_EPOCHS = 500
      
    print("開始訓練...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train")
        for blurry_images, original_images in train_loop:
            blurry_images, original_images = blurry_images.to(device), original_images.to(device)

            optimizer.zero_grad()

            outputs = model(blurry_images)
            loss = MAE_criterion(outputs, original_images)+MSE_criterion(outputs, original_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val")
            for blurry_images, original_images in val_loop:
                blurry_images, original_images = blurry_images.to(device), original_images.to(device)
                
                outputs = model(blurry_images)
                loss = MAE_criterion(outputs, original_images)+MSE_criterion(outputs, original_images)
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, 訓練損失: {avg_train_loss:.4f}, 驗證損失: {avg_val_loss:.4f}")
        loss_curve.append(avg_train_loss)
        
    stop=ti.time()
    time_spend=stop-start
    hour=int(time_spend/3600)
    minute=int((time_spend-3600*hour)/60)
    second=time_spend-3600*hour-60*minute
    plt.figure()
    plt.title("loss curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(loss_curve)
    print("訓練完成！")
    print(f"訓練時間:{hour}小時{minute}分{second}秒")

    torch.save(model.state_dict(), '降噪模型_1001.pth')
if __name__ == '__main__':
    main()
    
'''
訓練時間:4小時8分48.02795195579529秒  1001
訓練時間:21小時25分38.73669457435608秒  0930
'''
'''
#-----------------------------------------------------------
# 使用範例
#-----------------------------------------------------------
if __name__ == '__main__':
    # 假設你的SIDD數據集放在這個路徑下
    DATA_DIR = 'path/to/SIDD_data_directory/Data'
    
    # 定義圖像轉換
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)), # 統一圖像尺寸
        transforms.ToTensor(),
    ])
    
    # 實例化資料集
    sidd_dataset = SBDDenoisingDataset(data_dir=DATA_DIR, transform=image_transform)

    # 建立 DataLoader
    BATCH_SIZE = 4
    sidd_dataloader = DataLoader(sidd_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"資料集中的場景實例總數：{len(sidd_dataset)}")
    
    # 測試資料載入
    for noisy_images, clean_images in sidd_dataloader:
        print(f"帶噪聲圖像批次的形狀：{noisy_images.shape}")
        print(f"乾淨圖像批次的形狀：{clean_images.shape}")
        break
'''
    
'''
#模型定義

class ConvResBlock(nn.Module):
    def __init__(self,channel):
        super(ConvResBlock, self).__init__()
        self.convlayer1=nn.Conv2d(channel,channel,kernel_size=11,padding=5)
        self.convlayer2=nn.Conv2d(channel,channel, kernel_size=5,padding=2)
        self.convlayer3=nn.Conv2d(channel,channel, kernel_size=5,padding=2)
    def forward(self,x):
        output=F.leaky_relu(self.convlayer1(x))
        output=F.leaky_relu(self.convlayer2(output))
        output=self.convlayer3(output)
        output=output+x
        return output
#設計成自編碼器的形狀以濾除雜訊與模糊    
class DenoiseBlock(nn.Module):
        def __init__(self,encode_channel,#編碼器通道
                     bottleneck_channel,#瓶頸層通道
                     decode_channel,#解碼器通道
                     encode_depth,#編碼器深度
                     bottleneck_depth,#瓶頸層深度
                     decode_depth):#解碼器深度
            super(DenoiseBlock, self).__init__()
            self.adapterlayer1=nn.Conv2d(3,encode_channel,kernel_size=3,padding=1)#轉接層，調節通道大小以連接各個殘差塊
            self.EncodeBlock=nn.Sequential(*[ConvResBlock(encode_channel) for _ in range(encode_depth)])#編碼層，去掉多餘資訊
            self.adapterlayer2=nn.Conv2d(encode_channel,bottleneck_channel,kernel_size=3,padding=1)#轉接層，調節通道大小以連接各個殘差塊
            self.BottleneckBlock=nn.Sequential(*[ConvResBlock(bottleneck_channel) for _ in range(bottleneck_depth)])#瓶頸層，儲存濃縮資訊
            self.adapterlayer3=nn.Conv2d(bottleneck_channel,decode_channel,kernel_size=3,padding=1)#轉接層，調節通道大小以連接各個殘差塊
            self.DecoserBlock=nn.Sequential(*[ConvResBlock(decode_channel) for _ in range(decode_depth)])#解碼層，重建圖片
            self.RGBadapterlayer=nn.Conv2d(decode_channel,3,kernel_size=3,padding=1)#轉接層，調節通道大小以連接各個殘差塊
        def forward(self,x):
            output=F.leaky_relu(self.adapterlayer1(x))
            output=self.EncodeBlock(output)#因使用nn.Sepuential定義此區塊，所以是屬性是函式，所以可以直接呼叫
            output=F.leaky_relu(self.adapterlayer2(output))
            output=self.BottleneckBlock(output)
            output=F.leaky_relu(self.adapterlayer3(output))
            output=self.DecoserBlock(output)
            output=self.RGBadapterlayer(output)
            return output
                
'''         
        
        
        
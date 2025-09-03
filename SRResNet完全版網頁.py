# ==============================================================================
# Gradio 超解析度應用程式 (V3 - 整合無縫拼接)
# ==============================================================================
# 描述：
# 最終版本。整合了最先進的無縫拼接（平滑權重）切塊推論邏輯，
# 以在處理大尺寸圖片時，提供最佳的視覺效果，避免產生格子感。
# ==============================================================================

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm
import gradio as gr
import sys
import os
import numpy as np

# --- 1. 參數設定 ---
MODEL_PATH = 'srresnet_final.pth'
NUM_BLOCKS = 2
NUM_CHANNELS = 48
UPSCALE_FACTOR = 2
TILE_SIZE = 128
OVERLAP = 32 # 使用較大的重疊區域以獲得更好的平滑效果
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 模型定義 (與訓練時完全相同) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels) 
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels) 
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + residual
        return out

class SRResNet_small(nn.Module):
    def __init__(self, num_channels_in=3, num_blocks=4, num_channels_internal=64):
        super(SRResNet_small, self).__init__()
        self.conv_in = nn.Conv2d(num_channels_in, num_channels_internal, kernel_size=9, padding=4)
        self.relu_in = nn.PReLU()
        self.blocks = nn.Sequential(*[ResidualBlock(num_channels_internal) for _ in range(num_blocks)])
        self.conv_mid = nn.Conv2d(num_channels_internal, num_channels_internal, kernel_size=3, padding=1)
        self.norm_mid = nn.InstanceNorm2d(num_channels_internal)
        self.conv_out = nn.Conv2d(num_channels_internal, num_channels_in, kernel_size=9, padding=4)
    def forward(self, x):
        x_in = self.relu_in(self.conv_in(x))
        x_res = self.blocks(x_in)
        x_mid = self.norm_mid(self.conv_mid(x_res))
        x = x_in + x_mid
        x = self.conv_out(x)
        return x

def tensor_to_pil(tensor):
    return ToPILImage()(torch.clamp(tensor, 0, 1).cpu())

# --- 3. 載入模型 (在腳本啟動時只載入一次) ---
print(f"將使用 {DEVICE} 進行推論...")
try:
    model = SRResNet_small(num_blocks=NUM_BLOCKS, num_channels_internal=NUM_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型載入成功。")
except Exception as e:
    print(f"錯誤：模型載入失敗: {e}")
    model = None

# --- 4. 定義核心推論函式 (使用無縫拼接邏輯) ---
def super_resolve(input_image: Image.Image) -> Image.Image:
    if model is None: raise gr.Error("模型未能成功載入，請檢查終端錯誤訊息。")
    if input_image is None: raise gr.Error("請先上傳一張圖片。")
        
    input_image = input_image.convert("RGB")
    print(f"收到一張圖片，尺寸: {input_image.size}")
    
    # 預處理
    w, h = input_image.size
    target_w, target_h = w * UPSCALE_FACTOR, h * UPSCALE_FACTOR
    input_bicubic = input_image.resize((target_w, target_h), Image.BICUBIC)
    input_tensor = ToTensor()(input_bicubic).unsqueeze(0) # 保留在 CPU
    
    # 執行帶有平滑權重的切塊推論
    with torch.no_grad():
        c, h, w = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
        output_tensor = torch.zeros_like(input_tensor)
        weight_map = torch.zeros_like(input_tensor)
        
        # 產生一個中心權重高、向邊緣線性遞減的權重模板
        tile_weight = np.ones((TILE_SIZE, TILE_SIZE), dtype=np.float32)
        half_overlap = OVERLAP // 2
        for i in range(half_overlap):
            fade = (i + 1) / (half_overlap + 1)
            tile_weight[i, :] *= fade
            tile_weight[-1-i, :] *= fade
            tile_weight[:, i] *= fade
            tile_weight[:, -1-i] *= fade
        tile_weight_tensor = torch.from_numpy(tile_weight).unsqueeze(0).unsqueeze(0)

        stride = TILE_SIZE - OVERLAP
        pbar = tqdm(total=len(range(0, h, stride)) * len(range(0, w, stride)), desc="處理圖塊 (平滑拼接)")
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                h_end, w_end = min(i + TILE_SIZE, h), min(j + TILE_SIZE, w)
                h_start, w_start = h_end - TILE_SIZE, w_end - TILE_SIZE
                
                tile_input = input_tensor[:, :, h_start:h_end, w_start:w_end].to(DEVICE)
                
                tile_output = model(tile_input)
                
                output_tensor[:, :, h_start:h_end, w_start:w_end] += tile_output.cpu() * tile_weight_tensor
                weight_map[:, :, h_start:h_end, w_start:w_end] += tile_weight_tensor
                pbar.update(1)
        
        pbar.close()
        final_output = output_tensor / (weight_map + 1e-8)

    output_image_pil = tensor_to_pil(final_output.squeeze(0))
    print(f"處理完成，輸出尺寸: {output_image_pil.size}")
    return output_image_pil

# --- 5. 建立並啟動 Gradio 介面 ---
if model is not None:
    print("\n正在啟動 Gradio 介面...")
    example_path = 'input.jpg'
    if not os.path.exists(example_path): example_path = None
        
    iface = gr.Interface(
        fn=super_resolve,
        inputs=gr.Image(type="pil", label="請上傳低解析度圖片"),
        outputs=gr.Image(type="pil", label="超解析度結果"),
        title="SRResNet 超解析度模型 ",
        description="一個使用 PyTorch 和 Gradio 建立的影像超解析度工具。採用了切塊推論與平滑拼接技術，可處理任意尺寸的大圖並避免格子感。",
        examples=[[example_path]] if example_path else None
    )
    iface.launch()
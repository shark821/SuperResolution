# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 00:37:17 2025

@author: 周俊宇
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 20:29:41 2025

@author: 周俊宇
"""

# test_gradio.py

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import gradio as gr
import os

# 從 train.py 引入模型和資料集類別
from 文字銳利化 import DenoiseBlock, ConvResBlock

def load_model():
    """載入模型和權重。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型參數，必須與訓練時的參數一致
    encode_channel = 12
    bottleneck_channel = 10
    decode_channel = 12
    encode_depth = 15
    bottleneck_depth = 5
    decode_depth = 15
    
    model = DenoiseBlock(encode_channel, bottleneck_channel, decode_channel,
                         encode_depth, bottleneck_depth, decode_depth).to(device)
    
    MODEL_PATH = '文字銳利化模型_1001.pth' 
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型權重檔案: {MODEL_PATH}，請先運行訓練。")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

# 全局變數，只在程式啟動時載入一次模型
model, device = load_model()

def deblur_image(input_image):
    """
    使用模型對輸入圖像進行去模糊處理。
    """
    if input_image is None:
        return Image.new('RGB', (256, 256), 'black')

    # 將 PIL 圖像轉為張量
    input_tensor = ToTensor()(input_image).unsqueeze(0).to(device)
    
    # 執行模型推論
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 將輸出張量轉為 PIL 圖像
    output_image = ToPILImage()(output_tensor.cpu().squeeze(0))
    
    return output_image

# 創建 Gradio 介面
iface = gr.Interface(
    fn=deblur_image,
    inputs=gr.Image(label="貼上模糊圖片", type="pil"),
    outputs=gr.Image(label="去模糊輸出"),
    title="文字銳利化模型測試",
    description="請在下方區域貼上你的模糊圖片，模型將會自動進行去模糊處理。"
)

iface.launch()

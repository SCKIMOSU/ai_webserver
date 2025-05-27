import torch
import torchvision.transforms as transforms

import os


import torch
import os
from .model_arch import SimpleModel  # 클래스 정의 분리

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pt')

    model = SimpleModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # [1, C, H, W] 형태로 변환
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return int(predicted.item())

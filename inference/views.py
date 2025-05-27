from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from PIL import Image
import torch
import torchvision.transforms as transforms

from .ai_model.utils import load_model, predict_image  # 유틸 함수 분리 (아래에 예시 포함)

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        # 파일이 요청에 포함되어 있는지 확인
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['file']
        try:
            image = Image.open(image_file).convert('RGB')
        except Exception as e:
            return Response({'error': 'Invalid image file'}, status=status.HTTP_400_BAD_REQUEST)

        # 모델 로딩 및 예측
        model = load_model()
        prediction = predict_image(model, image)

        return Response({'prediction': prediction}, status=status.HTTP_200_OK)

    def get(self, request, *args, **kwargs):
        return Response(
            {'detail': 'GET method not allowed. Please POST an image file.'},
            status=status.HTTP_405_METHOD_NOT_ALLOWED
        )

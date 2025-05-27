# Django + AI 모델 + REST API

## ✅ 이미지에서 개를 분류하는 웹 서버 (AI 기능 내장) : 스켈러튼 코드

### 1. 프로젝트 개요

- **백엔드**: Django + Django REST Framework
- **AI 기능**: PyTorch or TensorFlow 모델을 통해 이미지 분석
- **API 기능**: 사용자로부터 이미지를 업로드받아 예측 결과 반환
- **프론트엔드 (선택)**: React 또는 Django 템플릿

---

### 2. 전체 디렉토리 구조 예시

```
ai_webserver/
├── manage.py
├── requirements.txt
├── ai_webserver/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── inference/
│   ├── __init__.py
│   ├── views.py       # AI 모델 예측 로직
│   ├── models.py      # Django 모델 (DB용)
│   ├── urls.py
│   └── ai_model/
│       ├── model.pt   # PyTorch 모델 파일
│       └── utils.py   # 전처리 / 후처리 함수
└── media/

```

---

### 3. 핵심 코드

### requirements.txt

```
Django==4.2
djangorestframework
Pillow
torch      # 또는 tensorflow

```

---

### settings.py 설정 추가

```python
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

```

---

### urls.py (project level)

```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('inference.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

```

---

### inference/urls.py

```python
from django.urls import path
from .views import PredictView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
]

```

---

### inference/views.py

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from PIL import Image
import torch
import torchvision.transforms as transforms
from .ai_model.utils import load_model, predict_image

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        image_file = request.FILES['file']
        image = Image.open(image_file).convert('RGB')
        model = load_model()
        prediction = predict_image(model, image)
        return Response({'prediction': prediction})

```

---

### inference/ai_model/utils.py

```python
import torch
from torchvision import transforms

def load_model():
    model = torch.load('inference/ai_model/model.pt', map_location='cpu')
    model.eval()
    return model

def predict_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # Batch size 1
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    return int(predicted.item())

```

---

### 4. 실행 방법

```bash
python manage.py migrate
python manage.py runserver

```

### 5. 테스트 (Postman 또는 curl)

```bash
curl -X POST -F 'file=@dog.jpg' http://localhost:8000/api/predict/

```

---

### ✅ AI  + 장고 아이디어

- 이미지 분석 (고양이/개 분류, 얼굴 감정 인식)
- 자연어 처리 (입력 문장 요약, 감정 분석)
- 음성 인식
- 추천 시스템 (사용자 기반 콘텐츠 추천)
- PDF에서 텍스트 추출 후 요약

---

## AI 모델 예측용 `PredictView` **코드**

- 이 코드는 이미지 파일을 받아 PyTorch 모델을 통해 예측 결과를 반환하는 API 구성.

---

## ✅ `inference/views.py`

```python
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

```

---

---

## ✅ 전체 클래스 흐름 설명: 이미지 예측 API

```python
class PredictView(APIView):
    ...
    def post(self, request, *args, **kwargs):
        ...

```

- `APIView`를 상속받아 `POST` 요청과 `GET` 요청에 대해 각각 동작을 정의.
- 특히 이 코드는 **이미지 파일을 받아 AI 모델로 분류하고 결과를 반환하는 REST API**.

---

## ✅ 1. POST 요청 처리

```python
def post(self, request, *args, **kwargs):

```

- 사용자가 API로 이미지를 전송할 때 사용하는 **HTTP POST** 메서드 정의.
- `request.FILES`에 첨부된 파일 데이터를 담고 있음.

---

### 🔸 1-1. 파일이 요청에 포함됐는지 확인

```python
if 'file' not in request.FILES:
    return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

```

- `request.FILES`는 form-data 형식으로 전송된 파일을 담는 딕셔너리.
- `'file'`이라는 키가 없으면 에러 응답 반환

📌 **Postman**에서 key를 정확히 `file`로 지정해야 함.

---

### 🔸 1-2. 파일을 이미지로 열기 + RGB 변환

```python
image_file = request.FILES['file']
try:
    image = Image.open(image_file).convert('RGB')
except Exception as e:
    return Response({'error': 'Invalid image file'}, status=status.HTTP_400_BAD_REQUEST)

```

- 파일을 **PIL 이미지 객체**로 변환
- `.convert('RGB')`: PNG 등으로 인해 채널 수가 다를 수 있으므로 3채널로 강제 통일
- 이미지가 손상되어 있거나 비정상 파일이면 에러 반환

---

### 🔸 1-3. 모델 로딩 및 예측 수행

```python
model = load_model()
prediction = predict_image(model, image)

```

- `load_model()`: 저장된 PyTorch 모델 불러오기 (`.pt` 파일)
- `predict_image(model, image)`: 전처리 + 추론 + 결과 반환

> prediction은 클래스 index (예: 0 or 1)
> 

---

### 🔸 1-4. 최종 응답 반환

```python
return Response({'prediction': prediction}, status=status.HTTP_200_OK)

```

- 예: `{ "prediction": 1 }`
- 필요 시 label 이름도 추가 가능: `{ "prediction": 1, "label": "cat" }`

---

## ✅ 2. GET 요청 대응

```python
def get(self, request, *args, **kwargs):
    return Response(
        {'detail': 'GET method not allowed. Please POST an image file.'},
        status=status.HTTP_405_METHOD_NOT_ALLOWED
    )

```

- 브라우저에서 `/api/predict/` 주소를 GET 요청으로 접근할 경우
- DRF가 `GET`을 자동으로 렌더링하려다 실패하지 않도록 **명시적으로 거절 응답** 반환
- `405 Method Not Allowed`: RESTful한 처리 방식

---

## ✅ 정리 요약

| 단계 | 기능 |
| --- | --- |
| `request.FILES['file']` | 사용자가 보낸 이미지 파일 접근 |
| `Image.open(...).convert('RGB')` | PIL 이미지 객체로 변환 |
| `load_model()` | 저장된 모델 불러오기 |
| `predict_image(...)` | 전처리 + 추론 수행 |
| `Response(...)` | JSON 형식으로 결과 응답 |

---

## ✅ `inference/ai_model/utils.py`

```python
import torch
import torchvision.transforms as transforms

def load_model():
    model = torch.load('inference/ai_model/model.pt', map_location='cpu')
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

```

---

## ✅ 예측 결과

예를 들어 모델이 **강아지 vs 고양이**를 분류한다면:

```json
{
  "prediction": 0
}

```

이런 식으로 반환. (0: 고양이, 1: 강아지 등은 클라이언트가 해석)

---

## ✅ Postman 테스트 방법

- URL: `http://127.0.0.1:8000/api/predict/`
- Method: POST
- Body: `form-data`
    - key: `file`
    - type: File
    - value: [이미지 업로드]

---

## `predict_image()` 함수

---

## 🔍 전체 코드

```python
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

```

---

## ✅  설명

### 1. 이미지 전처리 정의

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

```

- `transforms.Compose([...])`: 여러 전처리 과정을 순차적으로 적용.
- `transforms.Resize((224, 224))`: 입력 이미지를 모델 입력 크기인 **224x224**로 리사이즈.
    - ResNet, VGG 등 대부분의 CNN은 고정 크기 입력을 요구.
- `transforms.ToTensor()`:
    - PIL 이미지 → PyTorch Tensor 변환
    - 픽셀 값 정규화

---

---

## 🟡 PIL(Python Imaging Library) 이미지란?

- `PIL`은 Python에서 **이미지를 처리**할 수 있는 가장 기본적인 라이브러리.
- 현재는 `Pillow`라는 이름으로 유지보수됨. (`pip install pillow`)
- `PIL.Image.open("파일명.jpg")`으로 불러온 이미지는 `PIL.Image.Image` 클래스 객체.

### 예:

```python
from PIL import Image

img = Image.open("dog.jpg")
print(type(img))  # <class 'PIL.Image.Image'>

```

---

## 🟡 PIL 이미지의 주요 특성

| 속성 | 예시 | 설명 |
| --- | --- | --- |
| `.size` | `(width, height)` | 너비, 높이 |
| `.mode` | `"RGB"`, `"L"` 등 | 색상 모드 (RGB, 흑백 등) |
| `.format` | `"JPEG"`, `"PNG"` | 파일 포맷 |

---

## 🟢 PyTorch 모델 입력으로 사용하려면?

- PyTorch 모델은 입력으로 **Tensor** 형식의 이미지가 필요.
- 즉, PIL 이미지를 → Tensor로 변환해야 합니다.

---

## ✅ 변환: `transforms.ToTensor()`

```python
from torchvision import transforms

transform = transforms.ToTensor()
img_tensor = transform(pil_image)

```

### 내부적으로 변환되는 과정:

| 변환 단계 | 결과 형식 |
| --- | --- |
| PIL.Image (H, W, C) | → Tensor (C, H, W) |
| 픽셀 값: 0~255 | → float: 0.0 ~ 1.0 |

즉,

- 색상 채널이 맨 앞으로 이동
- 정규화됨
- 텐서 형태로 바뀜

---

### 예시:

```python
img = Image.open("dog.jpg").convert("RGB")
tensor = transforms.ToTensor()(img)
print(tensor.shape)  # torch.Size([3, H, W])
print(tensor.dtype)  # torch.float32

```

---

## 🧪 테스트 코드 예시

```python
from PIL import Image
from torchvision import transforms
import torch

# 1. 이미지 불러오기
image = Image.open("dog.jpg").convert("RGB")

# 2. 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. 전처리 적용
img_tensor = transform(image)
print(img_tensor.shape)     # torch.Size([3, 224, 224])
print(img_tensor.dtype)     # torch.float32
print(img_tensor.max())     # <= 1.0

```

---

## 🔁 정리

| 형식 | 의미 |
| --- | --- |
| PIL.Image | Python에서 읽어들인 원본 이미지 객체 |
| transforms.Resize | 크기 변환 |
| transforms.ToTensor | float32 텐서 변환 (정규화 + 채널순서 변경) |
| Tensor | PyTorch 모델 입력용 이미지 (C, H, W) |

---

### 2. 이미지 텐서화 및 배치 차원 추가

```python
img_tensor = transform(image).unsqueeze(0)

```

- `transform(image)`: PIL 이미지 → Tensor `[C, H, W]` 변환
- `.unsqueeze(0)`: 차원을 하나 추가해서 `[1, C, H, W]`로 변환
    
    → 모델은 일반적으로 **배치 단위 입력**을 요구하기 때문.
    
    예: `[3, 224, 224]` → `[1, 3, 224, 224]`
    

---

## ✅ 1`[C, H, W]`는 무엇인가?

- PyTorch에서 이미지 텐서는 **3차원 텐서**로 표현되며, 각 차원은 다음을 의미

| 축 | 의미 | 예시 |
| --- | --- | --- |
| `C` | **채널(Channel)** | 3 (RGB), 1 (Grayscale) |
| `H` | **높이(Height)** | 224, 256 등 |
| `W` | **너비(Width)** | 224, 256 등 |

즉, 이미지가 RGB 컬러이고 224x224 크기라면, PyTorch Tensor는:

```
torch.Size([3, 224, 224])

```

---

## ✅ 왜 `[C, H, W]` 형식인가?

- PyTorch의 CNN(합성곱 신경망)은 **채널이 먼저 오는 형식**(`channel-first`)을 기본으로 사용.
- 반면, NumPy나 PIL은 `(H, W, C)` 형식 (`channel-last`).

| 라이브러리 | 형식 |
| --- | --- |
| PyTorch | `[C, H, W]` |
| PIL | `(W, H)` |
| NumPy | `(H, W, C)` |

---

## ✅ 변환 과정

```python
from PIL import Image
from torchvision import transforms

img = Image.open("cat.jpg").convert("RGB")  # (W, H) + 3채널
transform = transforms.ToTensor()
tensor = transform(img)
print(tensor.shape)  # [3, H, W]

```

---

## ✅ 직접 보기: 각 채널의 의미

```python
print(tensor[0])  # Red 채널
print(tensor[1])  # Green 채널
print(tensor[2])  # Blue 채널

```

각 채널은 `[H, W]` 크기의 2차원 이미지.

---

## ✅ 5. 배치가 추가된 경우: `[N, C, H, W]`

- 모델에 입력하기 전에는 **배치 차원 N**도 추가.

```python
img_tensor = tensor.unsqueeze(0)  # [1, 3, 224, 224]

```

| 축 | 의미 |
| --- | --- |
| `N` | 배치 크기 (예: 1장, 32장) |
| `C` | 채널 |
| `H` | 높이 |
| `W` | 너비 |

---

## ✅ 시각화: `[C, H, W]` → `[H, W, C]`

```python
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# ToTensor의 역변환
img = F.to_pil_image(tensor)
plt.imshow(img)
plt.axis('off')
plt.show()

```

---

## 🔁 정리

| 항목 | 설명 |
| --- | --- |
| `[C, H, W]` | PyTorch 이미지 텐서 기본 형식 |
| `[3, 224, 224]` | RGB 이미지, 224x224 |
| `.unsqueeze(0)` | 배치 차원 추가 → `[1, 3, 224, 224]` |
| `.squeeze()` | 불필요한 차원 제거 |

---

### 3. 추론 시 gradient 계산 비활성화

```python
with torch.no_grad():

```

- 추론(predict)에서는 **역전파를 위한 gradient 계산이 불필요**하므로 이를 꺼서 메모리 사용량을 줄임.
- 속도가 빨라짐.

---

### 4. 모델 추론 실행

```python
output = model(img_tensor)

```

- `model()` 호출로 입력 텐서에 대한 예측값 생성
- `output`은 일반적으로 shape `[1, num_classes]`의 로짓(logits) 벡터
    
    예: `[[2.13, 0.88]]` → 클래스 0이 더 가능성 높음
    

---

- **처리된 이미지 텐서를 모델에 전달하여 예측값을 얻는 핵심 단계**

---

## ✅ 해설

| 항목 | 설명 |
| --- | --- |
| `img_tensor` | `[1, 3, 224, 224]` 형태의 입력 텐서 (배치 크기 1장) |
| `model` | 학습된 신경망 (예: `ResNet18`, `SimpleModel`, 등) |
| `model(img_tensor)` | **순전파(forward pass)** 실행: 입력 → 출력 |
| `output` | `[1, num_classes]` 형태의 로짓(logits) 텐서 |
| 예: `tensor([[2.35, 1.12]])` | 클래스별 점수 (softmax 이전의 raw 점수) |

---

## ✅ 고양이/강아지 분류

```python
output = model(img_tensor)
print(output)  # 예: tensor([[2.35, 1.12]])

```

- 첫 번째 값(2.35): 클래스 0 (예: "dog")
- 두 번째 값(1.12): 클래스 1 (예: "cat")
- 아직 softmax를 적용하지 않았지만, **값이 큰 쪽이 예측된 클래스.**

---

## ✅ 다음 단계: 가장 높은 클래스 선택

```python
_, predicted = torch.max(output, 1)
print(predicted.item())  # 0 or 1

```

- `torch.max(..., dim=1)`은 클래스 차원에서 가장 높은 값의 index를 반환
- 즉, `predicted`는 모델이 가장 가능성이 높다고 판단한 클래스

---

## ✅ 확률로 변환하려면?

```python
import torch.nn.functional as F

probs = F.softmax(output, dim=1)
print(probs)  # tensor([[0.81, 0.19]]) 처럼 확률 출력

```

- softmax는 로짓을 확률로 변환 (합: 1.0)
- 예: `[2.35, 1.12]` → `[0.81, 0.19]`

---

## 🔁 요약

| 단계 | 설명 |
| --- | --- |
| `model(img_tensor)` | 이미지에 대해 forward 연산 수행 |
| `output` | 모델의 raw 출력 (logits) |
| `torch.max(output, 1)` | 가장 높은 클래스 index 선택 |
| `F.softmax(output, 1)` | 확률 분포로 변환 (선택적) |

---

### 5. 가장 높은 점수를 가진 클래스 선택

```python
_, predicted = torch.max(output, 1)

```

- `torch.max(output, 1)`:
    - `output`에서 dim=1(클래스 차원) 기준으로 **최대값 위치(index)** 추출
    - 예: `[[2.13, 0.88]]` → `predicted = 0`
- `_`: 최대값 그 자체 (우리는 필요 없으므로 무시)

---

### 6. 정수 형태로 반환

```python
return int(predicted.item())

```

- `predicted`는 Tensor → `.item()`으로 Python 숫자로 변환
- `int(...)`: 정수형으로 명시적 캐스팅

---

## ✅ 최종 반환 값

예:

- 반환값이 `0` → 클래스 0 (예: "dog")
- 반환값이 `1` → 클래스 1 (예: "cat")

---

# Postman에서 API 테스트

- Postman에서 API 테스트를 하려는데 **서버가 구동되지 않는 경우**

---

## ✅ 1단계: Django 서버 실행 상태 확인

터미널에서 프로젝트 루트 디렉토리로 이동 후:

```bash
source venv/bin/activate      # 또는 .venv/bin/activate
python manage.py runserver

```

### 출력 예시 (정상일 경우):

```
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.

```

이 메시지가 보이지 않으면, 서버가 비정상.

---

## ✅ 2단계: 브라우저에서 기본 주소 확인

브라우저에서 다음 URL을 입력:

```
http://127.0.0.1:8000/

```

정상적인 Django 페이지가 뜨면 서버는 잘 실행 중.

---

## ✅ 3단계: Postman에서 요청 보내기

### 설정 요약:

- **URL**: `http://127.0.0.1:8000/api/predict/`
- **Method**: `POST`
- **Body**:
    - `form-data` 선택
    - Key: `file`
        
        Type: `File`
        
        Value: 이미지 파일 선택 (예: `dog.jpg`)
        

👉 아래 그림처럼 구성:

| Key | Type | Value |
| --- | --- | --- |
| file | File | (이미지 선택) |

---

## ✅ 4단계: API에서 POST만 허용했는지 확인

`views.py`에서 `PredictView` 클래스에 `get()`이 없는 경우, 브라우저에서 바로 접근하면 오류 발생.

**반드시 Postman으로 `POST` 요청**.

---

## ✅ 5단계: 서버 포트 충돌 확인

만약 이미 포트 8000번이 사용 중이라면:

```bash
python manage.py runserver 8080

```

 Postman에서는 다음 URL로 접근:

```
http://127.0.0.1:8080/api/predict/

```

---

## ✅ 6단계: 방화벽 / 네트워크 문제 (AWS/Lightsail 등 외부 접속 시)

만약 **외부 컴퓨터에서 Postman으로 접속 중**이라면,

- `python manage.py runserver 0.0.0.0:8000` 처럼 모든 IP에서 접근 가능하게 서버 실행.
- 그리고 AWS 보안그룹에서 포트 `8000`이 열려 있어야 함.

---

## 🛠️ 요약

| 항목 | 체크 |
| --- | --- |
| 가상환경 활성화 및 의존성 설치됨 | ✅ |
| `runserver` 정상 실행됨 | ✅ |
| `POST` 요청인지 확인 (GET 아님) | ✅ |
| Postman에서 `file` 키로 `form-data` 요청 | ✅ |
| 서버 주소와 포트 정확히 입력 | ✅ |
| 외부 접근 시: `0.0.0.0:8000` + AWS 보안그룹 설정 | ✅ |

---

## 에러 메시지 처리

```json
{
  "error": "No file provided"
}

```

- Django 서버가 **POST 요청은 받았지만**, `request.FILES` 안에 `"file"`이라는 키가 없어서 반환된 오류
    - **Postman 설정이 잘못되었거나 요청 형식이 틀렸다는 뜻.**

---

## ✅ 해결 방법: Postman에서 올바른 방식으로 이미지 전송

### 💡 아래와 같이 설정

### 1. **Method**: `POST`

### 2. **URL**:

```
http://127.0.0.1:8000/api/predict/

```

### 3. **Body 탭** 클릭 → `form-data` 선택

| Key | Type | Value |
| --- | --- | --- |
| file | File | (이미지 파일 첨부) |
- 반드시 key 이름이 `file`이어야 합니다 (**소문자**)
- Type은 `Text`가 아니라 **File**이어야 함.
- Value에 실제 이미지 파일을 첨부.

✅ 예시 화면 구성:

```
[Body]
[x] form-data

Key     |    Value           | Type
--------|--------------------|-------
file    |    cat.jpg         | File

```

---

## 🔎 서버 코드 확인

`PredictView`의 핵심 부분이 다음처럼 구성되어 있어야 함

```python
class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=400)

        image_file = request.FILES['file']
        ...

```

> request.FILES['file'] 는 multipart/form-data 요청에서만 채워지므로 반드시 Postman에서 form-data + File 전송 방식 사용
> 

---

## 🧪 테스트용 curl 예시

```bash
curl -X POST -F "file=@test.jpg" http://127.0.0.1:8000/api/predict/

```

---

---

## ✅ PyTorch 모델을 학습해서 `model.pt` 생성

### 🎯 분류 목적

- **이진 분류**: `dog` (클래스 0) vs `cat` (클래스 1)

---

## 🗂️ 1. 폴더 구조 예시 (ImageFolder용)

```
dataset/
├── train/
│   ├── dog/
│   │   ├── dog1.jpg ...
│   └── cat/
│       ├── cat1.jpg ...
├── val/
│   ├── dog/
│   └── cat/

```

---

## 🧠 2. 학습 코드 예시 (train_dog_cat.py)

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로
DATA_DIR = './dataset'
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

# 전처리
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 데이터셋
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# 모델: 사전학습된 resnet18 사용
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 이진 분류
model = model.to(DEVICE)

# 손실함수/최적화
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == y).sum().item()

    acc = correct / len(train_ds)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# 모델 저장
torch.save(model.state_dict(), "model.pt")
print("✅ model.pt 저장 완료")

```

---

## 📦 3. 모델 로딩 코드 (Django `load_model()`에서 사용)

```python
import torch
from torchvision import models
import torch.nn as nn

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("inference/ai_model/model.pt", map_location='cpu'))
    model.eval()
    return model

```

---

## 🎯 라벨 매핑 예시

```python
label_map = {0: "dog", 1: "cat"}

```

---

## 🔁 데이터가 없거나 학습이 어렵다면?

- `dummy model.pt` 생성
- 사전 학습된 강아지/고양이 모델 파일 공유 가능

## [`model.pt`](http://model.pt) 오류 메시지:

```
FileNotFoundError at /api/predict/
[Errno 2] No such file or directory: 'inference/ai_model/model.pt'

```

---

## ✅ 원인

- Django가 AI 모델 파일 `model.pt`을 불러오려 했지만 해당 경로에 **파일이 존재하지 않음**.

```python
torch.load('inference/ai_model/model.pt', map_location='cpu')

```

→ 이 경로는 프로젝트 기준 상대경로. 실제 해당 경로에 `.pt` 모델 파일이 있어야 함.

---

## ✅ 해결 방법

### ① 현재 디렉토리에서 파일 존재 확인

터미널에서 아래 명령으로 확인:

```bash
ls -l inference/ai_model/model.pt

```

### ❌ 없다면?

- 모델 파일이 아직 없는 것.

---

## ✅ 해결 방법 1: 예제 모델 생성하기 (임시로 동작 테스트용)

```python
# inference/ai_model/create_dummy_model.py
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(224 * 224 * 3, 2)  # 예: 이진 분류

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    model = SimpleModel()
    torch.save(model, "inference/ai_model/model.pt")
    print("Dummy model saved.")

```

### 실행:

```bash
python inference/ai_model/create_dummy_model.py

```

- 성공 시 `model.pt` 파일 생성.

---

## ✅ 해결 방법 2: 실제 학습된 모델 파일 준비

- 이미 학습한 PyTorch 모델이 있다면, 다음 형식으로 저장된 `.pt` 파일을 해당 위치로 복사:

```bash
cp ~/your_model_directory/model.pt inference/ai_model/

```

- 또는 Google Colab 등에서 저장 후 로컬로 옮김.

---

## ✅ load_model() 함수 확인 (utils.py)

```python
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

```

- 이렇게 절대경로 기반으로 수정하면 더 안전.

---

## ✅ 정리

| 상태 | 조치 |
| --- | --- |
| `model.pt` 없음 | 예제 모델 생성 또는 실제 모델 업로드 |
| 경로 문제 가능성 | `load_model()`에서 `os.path`로 절대경로 추천 |
| 테스트 목적이면 | dummy 모델 저장 코드 사용 |

---

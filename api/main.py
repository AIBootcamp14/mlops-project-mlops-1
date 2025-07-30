from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import sys

# src/features/feature_extractor.py 모듈 임포트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'features')))
from feature_extractor import WorkingFeatureExtractor

# 모델 및 특징 추출기 파일 경로 설정
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_classification_model.joblib')
extractor_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_extractor.joblib')

# 모델과 특징 추출기 로드
model = None
feature_extractor = None
try:
    model = joblib.load(model_path)
    feature_extractor = joblib.load(extractor_path)
    print("Model and Feature Extractor loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or Feature Extractor files not found at {model_path} or {extractor_path}. Ensure models are trained and saved.")
except Exception as e:
    print(f"Error loading model or feature extractor: {e}")

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "스팸 분류 API가 정상 작동 중입니다."}

@app.post("/predict")
def predict_spam(request: PredictionRequest):
    if model is None or feature_extractor is None:
        return {"error": "모델 또는 특징 추출기가 로드되지 않았습니다. 관리자에게 문의하세요."}

    input_text = request.text
    
    # 특징 추출기 사용하여 텍스트 특징 변환
    input_features = feature_extractor.transform([input_text]) 
    
    prediction = model.predict(input_features)[0]
    
    if prediction == 1:
        result = "스팸 (Spam)"
    else:
        result = "햄 (Ham)"

    return {"input_text": input_text, "prediction": result}
import joblib
import re
import nltk
from nltk.stem import PorterStemmer
import os
from fastapi import FastAPI
from pydantic import BaseModel

# NLTK 데이터 다운로드 (도커 이미지 빌드 단계에서 이미 실행되지만, 안전을 위해 추가)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# 텍스트 전처리 함수 (train_model.py에서 가져옴)
stemmer = PorterStemmer()
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 모델과 벡터라이저를 로드하는 함수
# 스크립트의 현재 위치(__file__)를 기준으로 상대 경로를 설정
def load_models():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '..', 'models', 'spam_classification_model.joblib')
    vectorizer_path = os.path.join(base_dir, '..', 'models', 'tfidf_vectorizer.joblib')
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("모델과 벡터라이저를 성공적으로 로드했습니다.")
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"에러: 파일 로드 실패 - {e}")
        print("모델 또는 벡터라이저 파일 경로를 확인해주세요.")
        return None, None
    except Exception as e:
        print(f"모델 로딩 중 예상치 못한 오류 발생: {e}")
        return None, None

# 앱 시작 시 모델 로드
model, vectorizer = load_models()

# 모델 로드 실패 시 앱 종료
if model is None or vectorizer is None:
    raise RuntimeError("모델 로드에 실패하여 애플리케이션을 시작할 수 없습니다.")

# 예측 요청을 위한 데이터 모델
class PredictionRequest(BaseModel):
    text: str

# 헬스 체크 엔드포인트
@app.get("/")
def read_root():
    return {"message": "스팸 분류 API가 실행 중입니다."}

# 예측 엔드포인트
@app.post("/predict")
def predict(request: PredictionRequest):
    # 1. 텍스트 전처리
    processed_text = preprocess_text(request.text)
    
    # 2. 벡터라이저를 사용해 텍스트를 벡터로 변환
    text_vectorized = vectorizer.transform([processed_text])
    
    # 3. 모델로 예측 수행
    prediction = model.predict(text_vectorized)[0]
    
    # 4. 예측 결과 반환
    result = "spam" if prediction == 1 else "ham"
    return {"prediction": result, "text": request.text}

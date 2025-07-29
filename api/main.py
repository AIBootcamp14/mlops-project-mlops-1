# 필요한 라이브러리 임포트
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="스팸 메시지 분류 API",
    description="입력된 메시지가 스팸인지 아닌지 분류하는 API입니다."
)

# 모델과 벡터라이저를 전역 변수로 선언 (애플리케이션 시작 시 로드)
# 초기에는 None으로 설정하고, 나중에 로드할 것임
tfidf_vectorizer = None
spam_classifier_model = None
porter_stemmer = None

# 모델과 벡터라이저 로드 함수
# 애플리케이션 시작 시 한 번만 실행됨
@app.on_event("startup")
async def load_model():
    global tfidf_vectorizer, spam_classifier_model, porter_stemmer

    # 모델 파일 경로 설정
    # 컨테이너 내부에서 파일 시스템 경로를 기준으로 함
    model_path = os.path.join(os.getcwd(), "models", "spam_classification_model.joblib")
    vectorizer_path = os.path.join(os.getcwd(), "models", "tfidf_vectorizer.joblib")

    # 파일 존재 여부 확인 (디버깅 용이)
    if not os.path.exists(model_path):
        raise RuntimeError(f"모델 파일이 존재하지 않습니다: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise RuntimeError(f"벡터라이저 파일이 존재하지 않습니다: {vectorizer_path}")

    try:
        # joblib을 사용하여 모델과 벡터라이저 로드
        spam_classifier_model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        porter_stemmer = PorterStemmer() # PorterStemmer 인스턴스 생성
        print("모델과 벡터라이저가 성공적으로 로드되었습니다.")
    except Exception as e:
        # 로드 실패 시 에러 발생
        raise RuntimeError(f"모델 또는 벡터라이저 로드 중 오류 발생: {e}")

# 텍스트 전처리 함수
# 모델 학습 시 사용했던 전처리 로직과 동일해야 함
def preprocess_text(text):
    # 소문자 변환
    text = text.lower()
    # 구두점 제거
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    # 불용어 제거 및 어간 추출
    words = text.split()
    filtered_words = [
        porter_stemmer.stem(word) for word in words if word not in stopwords.words('english')
    ]
    return ' '.join(filtered_words)

# API 요청 본문(Request Body)을 위한 Pydantic 모델 정의
# 클라이언트로부터 받을 메시지 형식을 정의
class Message(BaseModel):
    text: str # 입력 메시지 텍스트 필드

# 루트 엔드포인트 (API 상태 확인용)
@app.get("/")
async def read_root():
    return {"message": "스팸 분류 API가 정상 작동 중입니다."}

# 예측 엔드포인트
# 클라이언트로부터 메시지를 받아 스팸 여부를 예측
@app.post("/predict")
async def predict_spam(message: Message):
    if tfidf_vectorizer is None or spam_classifier_model is None:
        raise HTTPException(status_code=500, detail="모델이 아직 로드되지 않았습니다.")

    try:
        # 입력 메시지 전처리
        processed_text = preprocess_text(message.text)
        
        # TF-IDF 변환
        # transform은 2D 배열을 기대하므로 [processed_text] 형태로 전달
        text_vectorized = tfidf_vectorizer.transform([processed_text])
        
        # 모델 예측 (스팸: 1, 햄: 0)
        prediction = spam_classifier_model.predict(text_vectorized)[0]
        
        # 예측 결과에 따라 메시지 설정
        if prediction == 1:
            result = "스팸 (Spam)"
        else:
            result = "햄 (Ham)"
        
        return {"input_text": message.text, "prediction": result}
    except Exception as e:
        # 예측 중 오류 발생 시 에러 처리
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {e}")

# 애플리케이션을 직접 실행하기 위한 설정
# 이 파일이 직접 실행될 때만 uvicorn 서버를 시작
if __name__ == "__main__":
    import uvicorn
    # 8000번 포트에서 애플리케이션 실행
    # reload=True는 코드 변경 시 자동으로 서버를 재시작해 개발에 편리함
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
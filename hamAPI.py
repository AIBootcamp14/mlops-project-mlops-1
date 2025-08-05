from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# FastAPI 앱 생성
app = FastAPI()

# 모델과 벡터라이저 로드
model = joblib.load("models/spam_classifier_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# 입력 데이터 구조 정의
class InferenceInput(BaseModel):
    mail_sentences: str

# 추론 함수
def inference(data: np.ndarray):
    transformed = vectorizer.transform(data)
    prediction = model.predict(transformed)
    return prediction[0]  # 단일 예측 결과 반환

# API 엔드포인트
@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        input_text = input_data.mail_sentences
        print(f"📨 받은 문장: {input_text}")

        data = np.array([input_text])
        result = inference(data)

        print(f"📤 예측 결과: {result}")
        #return {"prediction": result}  # 예: spam 또는 ham
        return {"prediction": "spam" if result == 1 else "ham"}  # 예: spam 또는 ham

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

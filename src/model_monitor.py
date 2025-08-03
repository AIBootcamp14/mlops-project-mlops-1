import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import PorterStemmer
import re
import os # os 모듈을 추가하여 GITHUB_OUTPUT 환경 변수에 접근합니다.

# NLTK 데이터 다운로드 (로컬 실행을 위해)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 텍스트 전처리 함수
# train_model.py 스크립트와 동일한 함수를 사용해야 일관성을 유지할 수 있음
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

def monitor_model():
    logger.info("=== 모델 모니터링 시작 ===")
    
    # 기본값으로 재학습이 필요하지 않다고 설정
    retrain_needed = False

    # 1. 모델과 벡터라이저 로드
    model_path = 'models/spam_classification_model.joblib'
    vectorizer_path = 'models/tfidf_vectorizer.joblib'

    if not Path(model_path).exists() or not Path(vectorizer_path).exists():
        logger.error("❌ 모델 또는 벡터라이저 파일이 없습니다. 먼저 모델을 훈련시키세요.")
        # 재학습이 필요하다고 판단하여 새 워크플로우를 트리거하지 않도록 False로 설정
        return retrain_needed

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    logger.info("✅ 모델 및 벡터라이저 로드 완료.")

    # 2. 모니터링할 새로운 데이터 로드 (가상 경로)
    # 실제 운영 환경에서는 여기에 새로운 데이터가 들어올 것임
    # 지금은 테스트를 위해 가상의 데이터를 생성
    try:
        new_data_path = 'data/new_data/latest.csv'
        if not Path(new_data_path).exists():
            # 가상의 데이터 생성
            logger.warning("⚠️ 새로운 데이터 파일이 없어 가상의 데이터를 생성합니다.")
            sample_data = {
                'text': [
                    "WINNER! You have been selected for a free prize! Call now!",
                    "Hey, are you free for lunch tomorrow?",
                    "Congratulations! You've won a free iPhone. Click here.",
                    "Hi, let's catch up sometime next week.",
                    "Your account has been compromised. Click this link to reset password."
                ],
                'target': [1, 0, 1, 0, 1]  # 실제 데이터에서는 이 라벨이 없을 수 있음
            }
            new_df = pd.DataFrame(sample_data)
            # 가상 경로 폴더 생성
            Path('data/new_data').mkdir(parents=True, exist_ok=True)
            new_df.to_csv(new_data_path, index=False)
        
        df_new = pd.read_csv(new_data_path)
        logger.info(f"✅ 새로운 데이터 로드 완료: {df_new.shape}")
    except Exception as e:
        logger.error(f"❌ 새로운 데이터 로드 중 오류 발생: {e}")
        return retrain_needed

    # 3. 새로운 데이터 전처리 및 특징 추출
    logger.info("텍스트 전처리를 시작합니다.")
    df_new['processed_text'] = df_new['text'].apply(preprocess_text)
    X_new = vectorizer.transform(df_new['processed_text'])
    logger.info("✅ 데이터 전처리 및 특징 추출 완료.")

    # 4. 모델로 예측 수행
    logger.info("모델 예측을 수행합니다.")
    y_pred = model.predict(X_new)

    # 5. 모델 성능 평가 및 재학습 필요성 판단
    # 실제 운영 환경에서는 새로운 데이터에 정답(target)이 없을 수 있음.
    # 여기서는 예시를 위해 정답이 있다고 가정하고 정확도를 계산
    if 'target' in df_new.columns:
        y_true = df_new['target']
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f"모델 정확도: {accuracy:.4f}")

        # 재학습 임계값 설정
        performance_threshold = 0.8
        if accuracy < performance_threshold:
            logger.warning(f"⚠️ 모델 성능이 임계값({performance_threshold:.2f})보다 낮습니다. 재학습이 필요합니다!")
            retrain_needed = True # 재학습 필요
        else:
            logger.info("✅ 모델 성능이 양호합니다. 재학습이 필요하지 않습니다.")
    else:
        # 정답이 없는 경우, 데이터 드리프트 감지 로직을 추가해야 함
        logger.warning("⚠️ 새로운 데이터에 정답(target) 컬럼이 없어 성능을 평가할 수 없습니다. 데이터 드리프트를 별도로 모니터링해야 합니다.")
        # 이 예제에서는 일단 항상 False를 반환
        retrain_needed = False

    return retrain_needed

if __name__ == "__main__":
    retrain = monitor_model()
    # GITHUB_OUTPUT에 재학습 필요 여부 변수를 저장
    # 이 변수는 워크플로우에서 조건문으로 사용될 예정
    # ::set-output은 더 이상 사용되지 않으므로 GITHUB_OUTPUT을 사용합니다.
    output_path = os.getenv('GITHUB_OUTPUT')
    if output_path:
        with open(output_path, 'a') as f:
            f.write(f'retrain_needed={str(retrain).lower()}\n')
    else:
        # GITHUB_OUTPUT 변수가 없을 경우 (로컬 환경 등)
        print(f'retrain_needed={str(retrain).lower()}')

    if retrain:
        logger.info("🔥 재학습 워크플로우를 트리거합니다.")
    else:
        logger.info("✨ 모니터링 완료. 재학습은 필요 없습니다.")


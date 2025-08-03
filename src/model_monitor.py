import joblib
import pandas as pd
import logging
from sklearn.metrics import accuracy_score

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_new_data():
    """
    외부 시스템으로부터 새로운 데이터를 가져오는 것을 시뮬레이션합니다.
    실제 환경에서는 데이터베이스나 API에서 데이터를 로드하는 코드로 대체됩니다.
    """
    logging.info("✅ 새로운 데이터 로드 완료: (4, 2)")
    new_data = {
        'text': [
            "WINNER!! You have won a prize! Claim it now!",
            "Hello, hope you're having a great day.",
            "URGENT: Your account has been compromised. Click here to verify.",
            "Can we reschedule our meeting for tomorrow?"
        ],
        'label': ['spam', 'ham', 'spam', 'ham']
    }
    return pd.DataFrame(new_data)

def preprocess_data(data, vectorizer):
    """데이터를 전처리하고 특징을 추출합니다."""
    logging.info("텍스트 전처리를 시작합니다.")
    # 실제 환경에서는 더 복잡한 전처리 과정이 포함될 수 있습니다.
    X_new = vectorizer.transform(data['text'])
    y_new = data['label'].apply(lambda x: 1 if x == 'spam' else 0)
    logging.info("✅ 데이터 전처리 및 특징 추출 완료.")
    return X_new, y_new

def main():
    """모델 모니터링 및 재학습 결정 로직"""
    logging.info("=== 모델 모니터링 시작 ===")
    
    try:
        # 1. 최신 모델 및 벡터라이저 로드
        model = joblib.load('models/spam_classification_model.joblib')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        logging.info("✅ 모델 및 벡터라이저 로드 완료.")
        
        # 2. 새로운 데이터 가져오기
        new_data = fetch_new_data()

        # 3. 데이터 전처리 및 예측
        X_new, y_new = preprocess_data(new_data, vectorizer)
        logging.info("모델 예측을 수행합니다.")
        predictions = model.predict(X_new)

        # 4. 모델 성능 평가
        accuracy = accuracy_score(y_new, predictions)
        logging.info(f"모델 정확도: {accuracy:.4f}")

        # 5. 재학습 결정
        threshold = 0.80 # 성능 임계값
        if accuracy < threshold:
            logging.warning(f"⚠️ 모델 성능이 임계값({threshold})보다 낮습니다. 재학습이 필요합니다!")
            print("retrain_needed=true")
        else:
            logging.info("✨ 모델 성능이 양호합니다. 재학습은 필요하지 않습니다.")
            print("retrain_needed=false")
            
    except FileNotFoundError:
        logging.error("❌ 모델 파일을 찾을 수 없습니다. 최초 실행이거나 CI/CD 파이프라인에 문제가 있습니다.")
        print("retrain_needed=true")
    except Exception as e:
        logging.error(f"❌ 오류 발생: {e}")
        print("retrain_needed=true")

if __name__ == "__main__":
    main()
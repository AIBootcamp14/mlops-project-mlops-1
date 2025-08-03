import joblib
import pandas as pd
import logging
import os
from sklearn.metrics import accuracy_score
from data_validator import validate_data
from data_drift_detector import detect_data_drift # <-- 이 부분을 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_new_data():
    """
    외부 시스템으로부터 새로운 데이터를 가져오는 것을 시뮬레이션합니다.
    실제 환경에서는 데이터베이스나 API에서 데이터를 로드하는 코드로 대체됩니다.
    """
    logging.info("✅ new_data.csv 파일에서 데이터 로드")
    return pd.read_csv('new_data.csv')

def fetch_reference_data():
    """
    모델 학습에 사용된 원본 데이터를 가져옵니다.
    """
    logging.info("✅ 원본 훈련 데이터 로드")
    # 원본 데이터셋의 경로를 지정
    return pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

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
    
    retrain_needed = "false"
    
    try:
        # 1. 최신 모델 및 벡터라이저 로드
        model = joblib.load('models/spam_classification_model.joblib')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        logging.info("✅ 모델 및 벡터라이저 로드 완료.")
        
        # 2. 원본 데이터 및 새로운 데이터 가져오기
        reference_data = fetch_reference_data() # <-- 원본 데이터 로드
        new_data = fetch_new_data()
        
        # 3. 데이터 유효성 검사
        if not validate_data(new_data):
            logging.error("❌ 데이터 유효성 검사 실패. 재학습을 중단합니다.")
            print("retrain_needed=false")
            if os.getenv('GITHUB_OUTPUT'):
                with open(os.getenv('GITHUB_OUTPUT'), 'a') as f:
                    f.write(f"retrain_needed=false\n")
            return

        # 4. 데이터 드리프트 감지 <-- 이 부분을 추가
        is_drift_detected = detect_data_drift(reference_data, new_data)
        
        # 5. 데이터 전처리 및 예측
        X_new, y_new = preprocess_data(new_data, vectorizer)
        logging.info("모델 예측을 수행합니다.")
        predictions = model.predict(X_new)

        # 6. 모델 성능 평가
        accuracy = accuracy_score(y_new, predictions)
        logging.info(f"모델 정확도: {accuracy:.4f}")

        # 7. 재학습 결정
        threshold = 0.80 # 성능 임계값
        if accuracy < threshold or is_drift_detected: # <-- 드리프트 감지 결과도 포함
            if is_drift_detected:
                logging.warning(f"⚠️ 데이터 드리프트 감지! 재학습이 필요합니다!")
            if accuracy < threshold:
                logging.warning(f"⚠️ 모델 성능이 임계값({threshold})보다 낮습니다. 재학습이 필요합니다!")
            retrain_needed = "true"
        else:
            logging.info("✨ 모델 성능이 양호하고 데이터 드리프트가 감지되지 않았습니다. 재학습은 필요하지 않습니다.")
            
    except FileNotFoundError:
        logging.error("❌ 모델 파일을 찾을 수 없습니다. 최초 실행이거나 CI/CD 파이프라인에 문제가 있습니다.")
        retrain_needed = "true"
    except Exception as e:
        logging.error(f"❌ 오류 발생: {e}")
        retrain_needed = "true"

    # 워크플로우에 출력 변수 전달
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.getenv('GITHUB_OUTPUT'), 'a') as f:
            f.write(f"retrain_needed={retrain_needed}\n")
    else:
        print(f"retrain_needed={retrain_needed}")

if __name__ == "__main__":
    main()

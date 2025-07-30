import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os
import sys


# 예은님의 추가데이터로 트레인 모델 내용 변경

# src/features/feature_extractor.py 모듈을 임포트하기 위한 경로 추가
# 이 부분은 현재 스크립트의 위치에 따라 상대 경로를 조정해야 할 수 있어.
# 여기서는 프로젝트 루트(MLOPS-PROJECT-MLOPS-1)에서 실행된다고 가정하고 경로를 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'features')))
from feature_extractor import WorkingFeatureExtractor # WorkingFeatureExtractor 임포트

# 모델과 추출기 저장 경로
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_model():
    # 1. 전처리된 데이터 로드 (팀원이 생성한 파일)
    train_data_path = 'data/processed/train_data_latest.csv'
    test_data_path = 'data/processed/test_data_latest.csv'

    try:
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        print(f"훈련 데이터 로드 완료: {len(train_df)} 샘플")
        print(f"테스트 데이터 로드 완료: {len(test_df)} 샘플")
    except FileNotFoundError as e:
        print(f"에러: 전처리된 데이터 파일을 찾을 수 없습니다. '{train_data_path}' 또는 '{test_data_path}' 경로를 확인하세요.")
        print("데이터 전처리 파이프라인이 먼저 실행되어야 합니다.")
        sys.exit(1)
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        sys.exit(1)

    X_train_raw = train_df['text']
    y_train = train_df['target']
    X_test_raw = test_df['text']
    y_test = test_df['target']

    # 2. 특징 추출기 초기화 및 학습/변환
    # feature_extractor.py에서 만든 특징 추출기를 사용
    feature_extractor = WorkingFeatureExtractor()
    
    # 훈련 데이터로 특징 추출기 학습 및 변환
    X_train_features = feature_extractor.fit_transform(X_train_raw)
    
    # 테스트 데이터 변환 (학습된 추출기 사용)
    X_test_features = feature_extractor.transform(X_test_raw)

    print(f"훈련 데이터 특징 추출 완료: {X_train_features.shape}")
    print(f"테스트 데이터 특징 추출 완료: {X_test_features.shape}")

    # 3. 모델 학습
    print("모델 학습 시작...")
    model = MultinomialNB()
    model.fit(X_train_features, y_train)
    print("모델 학습 완료.")

    # 4. 모델 평가
    y_pred = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

    print(f"\n모델 정확도: {accuracy:.4f}")
    print("\n분류 리포트:")
    print(report)

    # 5. 모델과 특징 추출기 저장
    try:
        joblib.dump(model, os.path.join(MODEL_DIR, 'spam_classification_model.joblib'))
        joblib.dump(feature_extractor, os.path.join(MODEL_DIR, 'feature_extractor.joblib')) # TF-IDF 대신 feature_extractor 저장
        print(f"\n모델과 특징 추출기가 '{MODEL_DIR}' 폴더에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"모델 또는 특징 추출기 저장 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    train_and_save_model()
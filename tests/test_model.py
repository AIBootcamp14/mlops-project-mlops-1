import pytest
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# 테스트할 모델과 데이터 파일 경로 (수정)
MODEL_PATH = 'models/spam_classification_model.joblib'
VECTORIZER_PATH = 'models/tfidf_vectorizer.joblib'
TEST_DATA_PATH = 'data/spam.csv'  # 이 부분을 수정했어

# pytest 픽스처(fixture)를 사용해서 테스트에 필요한 객체를 미리 로드
@pytest.fixture(scope="session")
def trained_model():
    """학습된 모델 로드"""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"모델 파일이 존재하지 않습니다: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope="session")
def vectorizer():
    """벡터라이저 로드"""
    if not os.path.exists(VECTORIZER_PATH):
        pytest.fail(f"벡터라이저 파일이 존재하지 않습니다: {VECTORIZER_PATH}")
    return joblib.load(VECTORIZER_PATH)

@pytest.fixture(scope="session")
def test_data():
    """테스트 데이터 로드"""
    if not os.path.exists(TEST_DATA_PATH):
        pytest.fail(f"테스트 데이터 파일이 존재하지 않습니다: {TEST_DATA_PATH}")
    
    df = pd.read_csv(TEST_DATA_PATH, encoding='latin-1', sep=',', quotechar='"', on_bad_lines='skip')
    
    if 'target' not in df.columns or 'text' not in df.columns:
        if len(df.columns) >= 2:
            df.columns = ['target', 'text'] + list(df.columns[2:])
        else:
            pytest.fail("데이터 파일에 'target' 또는 'text' 컬럼이 없습니다.")
    
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].fillna('')

    return df

def test_model_accuracy(trained_model, vectorizer, test_data):
    """
    1. 모델이 정상적으로 로드되는지 확인
    2. 테스트 데이터로 예측을 수행하고 정확도를 확인
    3. 정확도가 특정 임계값(예: 0.9) 이상인지 검증
    """
    X_test_text = test_data['text']
    y_test = test_data['target']
    
    # 텍스트를 벡터로 변환
    X_test_vec = vectorizer.transform(X_test_text.fillna(''))
    
    # 예측 수행
    y_pred = trained_model.predict(X_test_vec)
    
    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n모델 정확도: {accuracy:.4f}")
    
    # 최소 정확도 임계값 설정
    assert accuracy >= 0.9, f"모델 정확도가 임계값(0.9)보다 낮습니다: {accuracy:.4f}"

def test_prediction_on_sample_text(trained_model, vectorizer):
    """
    샘플 텍스트를 사용해 예측이 제대로 되는지 확인
    """
    spam_text = ["WINNER! You have won a new iPhone. Claim now!"]
    ham_text = ["Hey, let's grab lunch tomorrow?"]
    
    # 텍스트 전처리 (여기에도 전처리 함수가 필요해 보여서 추가했어)
    stemmer = PorterStemmer()
    def preprocess_text_for_test(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = nltk.word_tokenize(text)
        words = [stemmer.stem(word) for word in words]
        return ' '.join(words)
    
    import re
    import nltk
    from nltk.stem import PorterStemmer

    processed_spam_text = [preprocess_text_for_test(t) for t in spam_text]
    processed_ham_text = [preprocess_text_for_test(t) for t in ham_text]

    # 스팸 텍스트 예측
    spam_vec = vectorizer.transform(processed_spam_text)
    spam_pred = trained_model.predict(spam_vec)[0]
    
    # 햄 텍스트 예측
    ham_vec = vectorizer.transform(processed_ham_text)
    ham_pred = trained_model.predict(ham_vec)[0]
    
    # 예측 결과 검증 (spam: 1, ham: 0)
    assert spam_pred == 1, "스팸 텍스트 예측 실패"
    assert ham_pred == 0, "햄 텍스트 예측 실패"

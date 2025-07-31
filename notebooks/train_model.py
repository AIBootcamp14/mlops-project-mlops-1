# 필요한 라이브러리 임포트
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import nltk
from nltk.stem import PorterStemmer
import re
import csv # csv 모듈 임포트

# NLTK 데이터 다운로드 (GitHub Actions 워크플로우에서 이미 다운로드하지만, 로컬 실행을 위해 포함)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger')

# 텍스트 전처리 함수
stemmer = PorterStemmer()

def preprocess_text(text):
    # 소문자 변환
    text = text.lower()
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    # 특수 문자 제거 (알파벳과 공백만 남김)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 토큰화 및 어간 추출
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def train_model():
    print("모델 학습을 시작합니다.")

    # 1. 데이터 로드 (data/spam.csv 파일 사용)
    # 현재 스크립트 위치에서 상대 경로로 data/spam.csv를 찾습니다.
    try:
        # sep=';', engine='python', quotechar='"', quoting=csv.QUOTE_NONE 옵션을 추가하여 파싱을 강화합니다.
        # QUOTE_NONE은 따옴표 처리를 무시합니다.
        df = pd.read_csv('data/spam.csv', encoding='latin-1', sep=';', engine='python', quoting=csv.QUOTE_NONE)
        print("데이터 로드 성공: data/spam.csv")
    except FileNotFoundError:
        print("에러: 'data/spam.csv' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        exit(1)

    # 필요한 컬럼만 선택하고 이름 변경
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # 'spam'을 1, 'ham'을 0으로 인코딩
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 2. 텍스트 전처리 적용
    print("텍스트 전처리를 시작합니다.")
    df['processed_text'] = df['text'].apply(preprocess_text)
    print("텍스트 전처리 완료.")

    # 3. 데이터 분할 (훈련 세트와 테스트 세트)
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )
    print(f"훈련 데이터 크기: {len(X_train)}")
    print(f"테스트 데이터 크기: {len(X_test)}")

    # 4. TF-IDF 벡터라이저 학습 및 변환
    print("TF-IDF 벡터라이저 학습을 시작합니다.")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # 상위 5000개 특징 사용
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print("TF-IDF 벡터라이저 학습 및 변환 완료.")

    # 5. 모델 학습 (나이브 베이즈)
    print("모델 학습을 시작합니다 (나이브 베이즈).")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("모델 학습 완료.")

    # 6. 모델 평가
    print("모델 평가를 시작합니다.")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"정확도 (Accuracy): {accuracy:.4f}")
    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall): {recall:.4f}")
    print(f"F1 점수 (F1 Score): {f1:.4f}")
    print("모델 평가 완료.")

    # 7. 모델과 벡터라이저 저장
    # models 폴더가 없으면 생성
    import os
    os.makedirs('models', exist_ok=True)

    model_path = 'models/spam_classifier_model.joblib'
    vectorizer_path = 'models/tfidf_vectorizer.joblib'

    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"모델 저장 완료: {model_path}")
    print(f"벡터라이저 저장 완료: {vectorizer_path}")

if __name__ == "__main__":
    train_model()

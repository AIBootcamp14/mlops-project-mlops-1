import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib # 모델 저장/로드
#import matplotlib.pyplot as plt # 시각화
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # 혼동 행렬
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# 데이터 로딩 및 초기 컬럼 처리

#데이터셋 로드 (인코딩 자동 감지 시도)
try:
    df = pd.read_csv('./hamspam_processed.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('./hamspam_processed.csv', encoding='latin-1')

# 불필요한 컬럼 제거 및 필요한 컬럼만 유지
if 'Unnamed: 0' in df.columns:
    df = df[['target', 'text', 'processed_text']].copy()
else:
    df = df[['target', 'text', 'processed_text']].copy() # 'target', 'text' 컬럼만 가정

# 또는 NaN 행 제거 (타겟이 같이 사라질 수 있으므로 주의!)
df = df.dropna()

# 데이터 미리보기 및 정보 확인
print("display df##########")
#print(df.head())
print(df.info())
print("display df##########")

# 데이터 분할 및 TF-IDF 벡터화

# 필요한 컬럼만 복사
df_processed = df[['target', 'processed_text']].copy()

# 타겟 결측치 행 제거 (견고성을 위해 다시 포함)
df_processed.dropna(subset=['target'], inplace=True)

# 학습/테스트 데이터 분할 (75:25)
X_train, X_test, y_train, y_test = train_test_split(df_processed['processed_text'], df_processed['target'], test_size=0.25, random_state=42)

# TF-IDF 벡터라이저 초기화
tfidf_vectorizer = TfidfVectorizer()

## 또는 NaN 행 제거 (타겟이 같이 사라질 수 있으므로 주의!)
#X_train = X_train.dropna()

# 학습 데이터에 TF-IDF 적용 및 변환
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# 테스트 데이터 변환
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 데이터셋 형태 출력
print("X_train_tfidf 형태:", X_train_tfidf.shape)
print("X_test_tfidf 형태:", X_test_tfidf.shape)
print("y_train 형태:", y_train.shape)
print("y_test 형태:", y_test.shape)

# 모델 학습

# Multinomial Naive Bayes 모델 초기화
nb_model = MultinomialNB()

# 모델 학습
nb_model.fit(X_train_tfidf, y_train)

print("모델 학습 완료!")

# 모델 평가

# 테스트 데이터 예측
y_pred = nb_model.predict(X_test_tfidf)

# 정확도 계산 및 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.4f}")

# 추가 평가 지표 계산 및 출력
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1 점수 (F1 Score): {f1:.4f}")

# 모델과 벡터라이저 저장

# 학습된 모델 저장
joblib.dump(nb_model, 'models/spam_classifier_model.joblib')

# TF-IDF 벡터라이저 저장
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')

print("모델과 TF-IDF 벡터라이저가 'models/' 폴더에 성공적으로 저장되었습니다.")



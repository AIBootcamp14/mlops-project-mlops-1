# src/features/feature_extractor.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import joblib

class WorkingFeatureExtractor:
    def __init__(self):
        self.setup_logging()
        self.vectorizers = {}
    
    def setup_logging(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_basic_features(self, texts):
        self.logger.info("기본 특징 추출 중...")
        features = pd.DataFrame()
        texts_clean = texts.fillna('')

        features['text_length'] = texts_clean.str.len()
        word_counts = texts_clean.str.split().str.len()
        features['word_count'] = word_counts.fillna(0)
        features['avg_word_length'] = features['text_length'] / features['word_count'].replace(0, 1)

        features['upper_count'] = texts_clean.str.count(r'[A-Z]')
        features['upper_ratio'] = features['upper_count'] / features['text_length'].replace(0, 1)

        features['digit_count'] = texts_clean.str.count(r'\d')
        features['digit_ratio'] = features['digit_count'] / features['text_length'].replace(0, 1)

        features['exclamation_count'] = texts_clean.apply(lambda x: str(x).count('!'))
        features['question_count'] = texts_clean.apply(lambda x: str(x).count('?'))
        features['at_count'] = texts_clean.apply(lambda x: str(x).count('@'))

        spam_words = ['free', 'win', 'call', 'click', 'urgent', 'now']
        for word in spam_words:
            features[f'{word}_count'] = texts_clean.str.lower().str.count(word)

        self.logger.info(f"기본 특징 {len(features.columns)}개 추출 완료")
        return features

    def extract_tfidf_features(self, texts, fit=True):
        self.logger.info("TF-IDF 특징 추출 중...")
        texts_clean = texts.fillna('')

        if fit:
            vectorizer = TfidfVectorizer(
                max_features=5508,  # 모델 학습 시 저장된 feature 수로 고정
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            features = vectorizer.fit_transform(texts_clean)
            self.vectorizers['tfidf'] = vectorizer
            self.logger.info(f"TF-IDF 벡터라이저 학습 완료: {features.shape}")
        else:
            vectorizer = self.vectorizers['tfidf']
            features = vectorizer.transform(texts_clean)
            self.logger.info(f"TF-IDF 특징 변환 완료: {features.shape}")

        return features

    def apply_dimensionality_reduction(self, features, fit=True):
        self.logger.info("차원 축소 적용 중...")

        if fit:
            svd = TruncatedSVD(n_components=100, random_state=42)
            reduced = svd.fit_transform(features)
            self.vectorizers['svd'] = svd
            self.logger.info(f"SVD 차원 축소 완료: {features.shape} -> {reduced.shape}")
        else:
            svd = self.vectorizers['svd']
            reduced = svd.transform(features)
            self.logger.info(f"SVD 변환 완료: {features.shape} -> {reduced.shape}")

        return reduced

    def extract_all_features(self, df, fit=True):
        self.logger.info("=== 전체 특징 추출 시작 ===")

        basic_features = self.extract_basic_features(df['text'])
        tfidf_features = self.extract_tfidf_features(df['text'], fit=fit)
        reduced_tfidf = self.apply_dimensionality_reduction(tfidf_features, fit=fit)

        basic_array = np.nan_to_num(basic_features.values)

        combined_features = np.hstack([basic_array, reduced_tfidf])

        self.logger.info(f"특징 결합 완료:")
        self.logger.info(f"  - 기본 특징: {basic_array.shape}")
        self.logger.info(f"  - 축소된 TF-IDF: {reduced_tfidf.shape}")
        self.logger.info(f"  - 결합된 특징: {combined_features.shape}")

        return combined_features, basic_features

    def save_vectorizers(self, save_path='models/feature_extractor.joblib'):
        self.logger.info(f"벡터라이저 저장 중: {save_path}")
        joblib.dump(self.vectorizers, save_path)
        self.logger.info("벡터라이저 저장 완료")

    def load_vectorizers(self, load_path='models/feature_extractor.joblib'):
        self.logger.info(f"벡터라이저 로딩 중: {load_path}")
        self.vectorizers = joblib.load(load_path)
        self.logger.info("벡터라이저 로딩 완료")

def main():
    print("🚀 간단한 특징 추출기 시작!")
    extractor = WorkingFeatureExtractor()

    train_path = 'data/processed/train_data_latest.csv'
    test_path = 'data/processed/test_data_latest.csv'

    if not Path(train_path).exists() or not Path(test_path).exists():
        print("❌ 전처리된 데이터 파일이 없습니다.")
        return

    print("📂 데이터 로딩 중...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"✅ 훈련 데이터: {train_df.shape}")
    print(f"✅ 테스트 데이터: {test_df.shape}")

    print("\n🔧 훈련 데이터 특징 추출 중...")
    X_train, train_basic = extractor.extract_all_features(train_df, fit=True)

    print("\n🔧 테스트 데이터 특징 변환 중...")
    X_test, test_basic = extractor.extract_all_features(test_df, fit=False)

    print("\n💾 특징 데이터 저장 중...")
    features_dir = Path('data/processed/features')
    features_dir.mkdir(exist_ok=True)

    np.save(features_dir / 'X_train_features.npy', X_train)
    np.save(features_dir / 'X_test_features.npy', X_test)
    np.save(features_dir / 'y_train.npy', train_df['target'].values)
    np.save(features_dir / 'y_test.npy', test_df['target'].values)

    train_basic.to_csv(features_dir / 'train_basic_features.csv', index=False)
    test_basic.to_csv(features_dir / 'test_basic_features.csv', index=False)

    extractor.save_vectorizers()

    print("\n" + "="*60)
    print("🎉 특징 추출 완료!")
    print("="*60)
    print(f"📈 훈련 특징: {X_train.shape}")
    print(f"📈 테스트 특징: {X_test.shape}")
    print(f"📈 기본 특징: {len(train_basic.columns)}개")
    print("\n🔤 추출된 기본 특징:")
    for i, feature in enumerate(train_basic.columns, 1):
        print(f"  {i:2d}. {feature}")

if __name__ == "__main__":
    main()

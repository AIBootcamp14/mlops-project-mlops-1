# data/scripts/preprocess_data.py
import pandas as pd
import re
import os
import logging
from datetime import datetime
from pathlib import Path
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

# NLTK 데이터 다운로드 (필요시)
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SimpleDataPreprocessor:
    """간단한 데이터 전처리 자동화 클래스"""
    
    def __init__(self):
        self.setup_logging()
        self.stemmer = PorterStemmer()
        self.setup_directories()
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """필요한 디렉토리 생성"""
        directories = ['data/raw', 'data/processed', 'logs']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """데이터 로딩 - 노트북과 동일한 방식"""
        self.logger.info("데이터 로딩 시작")
        
        data_path = 'data/spam.csv'
        
        # 인코딩 자동 감지 시도 (노트북과 동일)
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
            self.logger.info("UTF-8 인코딩으로 데이터 로드 성공")
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding='latin-1')
            self.logger.info("Latin-1 인코딩으로 데이터 로드 성공")
        
        self.logger.info(f"로드된 데이터 형태: {df.shape}")
        return df
    
    def clean_columns(self, df):
        """불필요한 컬럼 제거 및 필요한 컬럼만 유지 - 노트북과 동일"""
        self.logger.info("컬럼 정리 시작")
        
        # 불필요한 컬럼 제거 및 필요한 컬럼만 유지
        if 'Unnamed: 0' in df.columns:
            df = df[['target', 'text']].copy()
        else:
            # 컬럼명이 다를 수 있으므로 유연하게 처리
            possible_target_cols = ['target', 'label', 'class', 'spam']
            possible_text_cols = ['text', 'message', 'email', 'content']
            
            target_col = None
            text_col = None
            
            for col in df.columns:
                if col.lower() in possible_target_cols:
                    target_col = col
                elif col.lower() in possible_text_cols:
                    text_col = col
            
            if target_col and text_col:
                df = df[[target_col, text_col]].copy()
                df.columns = ['target', 'text']  # 표준 컬럼명으로 변경
            else:
                # 기본값으로 처음 두 컬럼 사용
                df = df.iloc[:, :2].copy()
                df.columns = ['target', 'text']
        
        self.logger.info(f"정리된 데이터 형태: {df.shape}")
        self.logger.info(f"컬럼: {list(df.columns)}")
        
        return df
    
    def process_target(self, df):
        """타겟 값 매핑 및 결측치/타입 처리 - 노트북과 동일"""
        self.logger.info("타겟 값 처리 시작")
        
        # 타겟 값 매핑 ('ham':0, 'spam':1)
        df['target'] = df['target'].map({'ham': 0, 'spam': 1}).fillna(df['target'])
        
        # 매핑 실패 또는 NaN 값 행 제거
        before_count = len(df)
        df.dropna(subset=['target'], inplace=True)
        after_count = len(df)
        
        if before_count != after_count:
            self.logger.info(f"결측치 제거: {before_count - after_count}개 행 제거")
        
        # 타겟 컬럼 정수형으로 변환
        df['target'] = df['target'].astype(int)
        
        # 타겟 분포 확인
        target_counts = df['target'].value_counts()
        self.logger.info(f"타겟 분포: {target_counts.to_dict()}")
        
        return df
    
    def preprocess_text(self, text):
        """텍스트 전처리 함수 - 노트북과 완전히 동일"""
        if not isinstance(text, str):  # 문자열이 아니면 빈 문자열 반환
            return ""
        text = text.lower()  # 소문자 변환
        text = re.sub(r'[^a-z]', ' ', text)  # 알파벳 외 제거
        text = text.split()  # 단어 분리
        text = [self.stemmer.stem(word) for word in text]  # 어간 추출
        return ' '.join(text)
    
    def apply_text_preprocessing(self, df):
        """전처리 함수 적용하여 새 컬럼 생성 - 노트북과 동일"""
        self.logger.info("텍스트 전처리 시작")
        
        # 전처리 함수 적용하여 새 컬럼 생성
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # 전처리 결과 확인
        self.logger.info("텍스트 전처리 완료")
        self.logger.info("전처리 예시:")
        for i in range(min(3, len(df))):
            self.logger.info(f"원본: {df.iloc[i]['text'][:50]}...")
            self.logger.info(f"전처리: {df.iloc[i]['processed_text'][:50]}...")
            self.logger.info("---")
        
        return df
    
    def split_and_save_data(self, df):
        """데이터 분할 및 저장"""
        self.logger.info("데이터 분할 및 저장 시작")
        
        # 필요한 컬럼만 복사
        df_processed = df[['target', 'processed_text']].copy()
        
        # 타겟 결측치 행 제거 (견고성을 위해 다시 포함)
        df_processed.dropna(subset=['target'], inplace=True)
        
        # 학습/테스트 데이터 분할 (75:25) - 노트북과 동일
        X_train, X_test, y_train, y_test = train_test_split(
            df_processed['processed_text'], 
            df_processed['target'], 
            test_size=0.25, 
            random_state=42
        )
        
        # 데이터셋 형태 출력
        self.logger.info(f"X_train 형태: {X_train.shape}")
        self.logger.info(f"X_test 형태: {X_test.shape}")
        self.logger.info(f"y_train 형태: {y_train.shape}")
        self.logger.info(f"y_test 형태: {y_test.shape}")
        
        # DataFrame 형태로 재구성
        train_df = pd.DataFrame({
            'text': X_train,
            'target': y_train
        })
        
        test_df = pd.DataFrame({
            'text': X_test,
            'target': y_test
        })
        
        # 전체 처리된 데이터도 저장
        full_processed_df = df_processed.copy()
        full_processed_df.columns = ['target', 'text']  # 컬럼명 통일
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_dir = Path('data/processed')
        
        # 타임스탬프 포함 파일
        train_df.to_csv(processed_dir / f'train_data_{timestamp}.csv', index=False)
        test_df.to_csv(processed_dir / f'test_data_{timestamp}.csv', index=False)
        full_processed_df.to_csv(processed_dir / f'full_processed_data_{timestamp}.csv', index=False)
        
        # 최신 파일 (다른 스크립트에서 사용하기 쉽도록)
        train_df.to_csv(processed_dir / 'train_data_latest.csv', index=False)
        test_df.to_csv(processed_dir / 'test_data_latest.csv', index=False)
        full_processed_df.to_csv(processed_dir / 'processed_data_latest.csv', index=False)
        
        self.logger.info("데이터 저장 완료:")
        self.logger.info(f"  - 훈련 데이터: {len(train_df)}개 샘플")
        self.logger.info(f"  - 테스트 데이터: {len(test_df)}개 샘플")
        self.logger.info(f"  - 전체 데이터: {len(full_processed_df)}개 샘플")
        
        return train_df, test_df, full_processed_df
    
    def generate_report(self, df, train_df, test_df):
        """처리 리포트 생성"""
        self.logger.info("처리 리포트 생성")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_data_shape': df.shape,
            'processed_data_shape': (len(train_df) + len(test_df), 2),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'target_distribution': {
                'train': train_df['target'].value_counts().to_dict(),
                'test': test_df['target'].value_counts().to_dict()
            }
        }
        
        # 리포트 저장
        import json
        with open('data/processed/preprocessing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # 리포트 출력
        self.logger.info("=== 전처리 리포트 ===")
        self.logger.info(f"처리 시간: {report['timestamp']}")
        self.logger.info(f"원본 데이터: {report['original_data_shape']}")
        self.logger.info(f"처리된 데이터: {report['processed_data_shape']}")
        self.logger.info(f"훈련 샘플: {report['train_samples']}")
        self.logger.info(f"테스트 샘플: {report['test_samples']}")
        self.logger.info(f"훈련 타겟 분포: {report['target_distribution']['train']}")
        self.logger.info(f"테스트 타겟 분포: {report['target_distribution']['test']}")
    
    def run_preprocessing_pipeline(self):
        """전체 전처리 파이프라인 실행"""
        try:
            self.logger.info("=== 데이터 전처리 파이프라인 시작 ===")
            
            # 1. 데이터 로딩
            df = self.load_data()
            
            # 2. 컬럼 정리
            df = self.clean_columns(df)
            
            # 3. 타겟 값 처리
            df = self.process_target(df)
            
            # 4. 텍스트 전처리
            df = self.apply_text_preprocessing(df)
            
            # 5. 데이터 분할 및 저장
            train_df, test_df, full_df = self.split_and_save_data(df)
            
            # 6. 리포트 생성
            self.generate_report(df, train_df, test_df)
            
            self.logger.info("=== 데이터 전처리 파이프라인 완료 ===")
            
            return {
                'status': 'success',
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'total_samples': len(full_df)
            }
            
        except Exception as e:
            self.logger.error(f"전처리 파이프라인 실행 중 오류: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    preprocessor = SimpleDataPreprocessor()
    result = preprocessor.run_preprocessing_pipeline()
    
    print("\n🎉 전처리 완료!")
    print(f"✅ 훈련 데이터: {result['train_samples']}개 샘플")
    print(f"✅ 테스트 데이터: {result['test_samples']}개 샘플")
    print(f"✅ 총 데이터: {result['total_samples']}개 샘플")
    print("\n📁 저장된 파일:")
    print("  - data/processed/train_data_latest.csv")
    print("  - data/processed/test_data_latest.csv")
    print("  - data/processed/processed_data_latest.csv")
    print("  - data/processed/preprocessing_report.json")


if __name__ == "__main__":
    main()
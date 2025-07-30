# data/scripts/validate_data.py
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

class DataValidator:
    """데이터 품질 검증 및 모니터링 클래스"""
    
    def __init__(self):
        self.setup_logging()
        self.validation_rules = self.load_validation_rules()
        
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'data_validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_validation_rules(self) -> Dict:
        """데이터 검증 규칙 정의"""
        return {
            'required_columns': ['target', 'text'],
            'target_values': [0, 1],
            'min_text_length': 1,
            'max_text_length': 5000,
            'min_dataset_size': 100,
            'target_balance_threshold': 0.1,  # 클래스 불균형 임계값 (10% 미만이면 경고)
            'duplicate_threshold': 0.05,  # 중복 데이터 임계값 (5% 이상이면 경고)
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 스키마 검증"""
        results = {
            'passed': True,
            'errors': [],
            'warnings': []
        }
        
        # 필수 컬럼 확인
        required_cols = self.validation_rules['required_columns']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            results['passed'] = False
            results['errors'].append(f"필수 컬럼 누락: {missing_cols}")
        
        # 데이터 타입 확인
        if 'target' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['target']):
                results['passed'] = False
                results['errors'].append("target 컬럼이 숫자형이 아닙니다")
        
        if 'text' in df.columns:
            if not pd.api.types.is_object_dtype(df['text']):
                results['warnings'].append("text 컬럼이 문자열 타입이 아닙니다")
        
        return results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 품질 검증"""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # 데이터셋 크기 확인
        if len(df) < self.validation_rules['min_dataset_size']:
            results['passed'] = False
            results['errors'].append(f"데이터셋 크기가 너무 작습니다: {len(df)} < {self.validation_rules['min_dataset_size']}")
        
        # 결측치 확인
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            results['warnings'].append(f"결측치 발견: {null_counts.to_dict()}")
        
        # target 값 범위 확인
        if 'target' in df.columns:
            invalid_targets = ~df['target'].isin(self.validation_rules['target_values'])
            if invalid_targets.sum() > 0:
                results['passed'] = False
                results['errors'].append(f"잘못된 target 값 {invalid_targets.sum()}개 발견")
            
            # 클래스 불균형 확인
            target_dist = df['target'].value_counts(normalize=True)
            min_class_ratio = target_dist.min()
            if min_class_ratio < self.validation_rules['target_balance_threshold']:
                results['warnings'].append(f"심각한 클래스 불균형: 최소 클래스 비율 {min_class_ratio:.3f}")
            
            results['metrics']['target_distribution'] = {int(k): int(v) for k, v in target_dist.items()}
        
        # 텍스트 길이 확인
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            
            # 빈 텍스트 확인
            empty_texts = text_lengths < self.validation_rules['min_text_length']
            if empty_texts.sum() > 0:
                results['warnings'].append(f"빈 텍스트 {empty_texts.sum()}개 발견")
            
            # 너무 긴 텍스트 확인
            long_texts = text_lengths > self.validation_rules['max_text_length']
            if long_texts.sum() > 0:
                results['warnings'].append(f"너무 긴 텍스트 {long_texts.sum()}개 발견")
            
            results['metrics']['text_length_stats'] = {
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'std': float(text_lengths.std())
            }
        
        # 중복 데이터 확인
        if 'text' in df.columns:
            duplicate_count = df['text'].duplicated().sum()
            duplicate_ratio = duplicate_count / len(df)
            
            if duplicate_ratio > self.validation_rules['duplicate_threshold']:
                results['warnings'].append(f"높은 중복 비율: {duplicate_ratio:.3f} ({duplicate_count}개)")
            
            results['metrics']['duplicate_ratio'] = float(duplicate_ratio)
            results['metrics']['duplicate_count'] = int(duplicate_count)
        
        return results
    
    def validate_preprocessing_consistency(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """전처리 일관성 검증"""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # 데이터 손실 확인
        original_size = len(original_df)
        processed_size = len(processed_df)
        data_loss_ratio = (original_size - processed_size) / original_size
        
        if data_loss_ratio > 0.1:  # 10% 이상 데이터 손실시 경고
            results['warnings'].append(f"높은 데이터 손실률: {data_loss_ratio:.3f}")
        
        results['metrics']['data_loss_ratio'] = float(data_loss_ratio)
        results['metrics']['original_size'] = int(original_size)
        results['metrics']['processed_size'] = int(processed_size)
        
        # target 분포 일관성 확인
        if 'target' in original_df.columns and 'target' in processed_df.columns:
            orig_dist = original_df['target'].value_counts(normalize=True).sort_index()
            proc_dist = processed_df['target'].value_counts(normalize=True).sort_index()
            
            dist_diff = abs(orig_dist - proc_dist).max()
            if dist_diff > 0.05:  # 5% 이상 분포 변화시 경고
                results['warnings'].append(f"target 분포 변화: 최대 차이 {dist_diff:.3f}")
            
            results['metrics']['target_distribution_change'] = float(dist_diff)
        
        return results
    
    def run_full_validation(self, data_path: str) -> Dict[str, Any]:
        """전체 검증 파이프라인 실행"""
        self.logger.info("=== 데이터 검증 시작 ===")
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'overall_passed': True,
            'schema_validation': {},
            'quality_validation': {},
            'consistency_validation': {},
            'summary': {}
        }
        
        try:
            # 데이터 로드
            df = pd.read_csv(data_path)
            self.logger.info(f"데이터 로드 완료: {df.shape}")
            
            # 1. 스키마 검증
            schema_results = self.validate_schema(df)
            validation_report['schema_validation'] = schema_results
            if not schema_results['passed']:
                validation_report['overall_passed'] = False
            
            # 2. 데이터 품질 검증
            quality_results = self.validate_data_quality(df)
            validation_report['quality_validation'] = quality_results
            if not quality_results['passed']:
                validation_report['overall_passed'] = False
            
            # 3. 원본 데이터와 비교 (전처리된 데이터인 경우)
            raw_data_path = 'data/spam.csv'
            if Path(raw_data_path).exists() and 'processed' in data_path:
                try:
                    original_df = pd.read_csv(raw_data_path, encoding='latin-1')
                    # 컬럼 정리 (전처리 스크립트와 동일한 방식)
                    if len(original_df.columns) >= 2:
                        original_df = original_df.iloc[:, :2].copy()
                        original_df.columns = ['target', 'text']
                        original_df['target'] = original_df['target'].map({'ham': 0, 'spam': 1}).fillna(original_df['target'])
                        original_df.dropna(subset=['target'], inplace=True)
                        original_df['target'] = original_df['target'].astype(int)
                    
                    consistency_results = self.validate_preprocessing_consistency(original_df, df)
                    validation_report['consistency_validation'] = consistency_results
                    if not consistency_results['passed']:
                        validation_report['overall_passed'] = False
                        
                except Exception as e:
                    self.logger.warning(f"원본 데이터와 비교 실패: {e}")
            
            # 4. 요약 생성
            total_errors = (len(schema_results.get('errors', [])) + 
                          len(quality_results.get('errors', [])) + 
                          len(validation_report['consistency_validation'].get('errors', [])))
            
            total_warnings = (len(schema_results.get('warnings', [])) + 
                            len(quality_results.get('warnings', [])) + 
                            len(validation_report['consistency_validation'].get('warnings', [])))
            
            validation_report['summary'] = {
                'total_errors': int(total_errors),
                'total_warnings': int(total_warnings),
                'data_shape': list(df.shape),  # tuple을 list로 변환
                'validation_status': 'PASSED' if validation_report['overall_passed'] else 'FAILED'
            }
            
            # 5. 리포트 저장
            report_path = Path('data/processed/validation_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, ensure_ascii=False)
            
            # 6. 로그 출력
            self.logger.info("=== 검증 결과 요약 ===")
            self.logger.info(f"상태: {validation_report['summary']['validation_status']}")
            self.logger.info(f"오류: {total_errors}개")
            self.logger.info(f"경고: {total_warnings}개")
            self.logger.info(f"데이터 크기: {df.shape}")
            
            if total_errors > 0:
                self.logger.error("❌ 데이터 검증 실패!")
                for section in ['schema_validation', 'quality_validation', 'consistency_validation']:
                    for error in validation_report.get(section, {}).get('errors', []):
                        self.logger.error(f"  - {error}")
            else:
                self.logger.info("✅ 데이터 검증 통과!")
            
            if total_warnings > 0:
                self.logger.warning(f"⚠️  {total_warnings}개 경고 발견:")
                for section in ['schema_validation', 'quality_validation', 'consistency_validation']:
                    for warning in validation_report.get(section, {}).get('warnings', []):
                        self.logger.warning(f"  - {warning}")
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"검증 중 오류 발생: {str(e)}")
            validation_report['overall_passed'] = False
            validation_report['error'] = str(e)
            return validation_report


def main():
    """메인 실행 함수"""
    validator = DataValidator()
    
    # 처리된 데이터 검증
    processed_files = [
        'data/processed/train_data_latest.csv',
        'data/processed/test_data_latest.csv',
        'data/processed/processed_data_latest.csv'
    ]
    
    results = {}
    for file_path in processed_files:
        if Path(file_path).exists():
            print(f"\n🔍 {file_path} 검증 중...")
            results[file_path] = validator.run_full_validation(file_path)
        else:
            print(f"❌ 파일 없음: {file_path}")
    
    # 전체 결과 요약
    print("\n" + "="*60)
    print("📊 전체 검증 결과 요약")
    print("="*60)
    
    for file_path, result in results.items():
        status = "✅ PASSED" if result['overall_passed'] else "❌ FAILED"
        errors = result['summary']['total_errors']
        warnings = result['summary']['total_warnings']
        print(f"{Path(file_path).name}: {status} (오류: {errors}, 경고: {warnings})")


if __name__ == "__main__":
    main()
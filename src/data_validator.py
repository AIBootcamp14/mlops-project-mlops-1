import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_data(data: pd.DataFrame) -> bool:
    """
    새로 들어온 데이터의 유효성을 검사합니다.
    - 필수 컬럼('text', 'label')이 존재하는지 확인합니다.
    - 'label' 컬럼의 값이 'spam' 또는 'ham'인지 확인합니다.
    """
    logging.info("데이터 유효성 검사를 시작합니다.")

    # 1. 필수 컬럼 확인
    required_columns = ['text', 'label']
    if not all(col in data.columns for col in required_columns):
        logging.error(f"❌ 데이터에 필수 컬럼이 누락되었습니다. 누락된 컬럼: {set(required_columns) - set(data.columns)}")
        return False
    
    # 2. 레이블 값 확인
    valid_labels = ['spam', 'ham']
    if not data['label'].isin(valid_labels).all():
        logging.error("❌ 'label' 컬럼에 유효하지 않은 값이 포함되어 있습니다. 유효한 값: 'spam', 'ham'")
        return False

    logging.info("✅ 데이터 유효성 검사 성공.")
    return True


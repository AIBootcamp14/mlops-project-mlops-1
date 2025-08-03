import pandas as pd
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
    """
    evidently 라이브러리를 사용하여 데이터 드리프트를 감지합니다.
    """
    logging.info("데이터 드리프트 감지를 시작합니다.")
    
    # evidently 리포트 생성 (DataDriftPreset 사용)
    data_drift_report = Report(metrics=[DataDriftPreset()])
    
    # 원본 데이터(reference_data)와 현재 데이터(current_data)를 비교하여 리포트 생성
    data_drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=None)
    
    # 리포트 결과에서 데이터 드리프트가 감지되었는지 확인
    report_json = data_drift_report.as_dict()
    dataset_drift_detected = report_json['metrics'][0]['result']['dataset_drift']
    
    if dataset_drift_detected:
        logging.warning("⚠️ 데이터 드리프트가 감지되었습니다!")
        return True
    else:
        logging.info("✨ 데이터 드리프트가 감지되지 않았습니다.")
        return False


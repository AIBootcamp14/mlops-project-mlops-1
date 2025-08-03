import pandas as pd
import os
import random
import csv # 이 부분을 추가

def generate_new_data():
    """
    new_data.csv 파일에 새로운 데이터를 추가합니다.
    실제 환경에서는 데이터베이스에 새로운 데이터가 쌓이는 것을 시뮬레이션합니다.
    """
    new_samples = [
        {"text": "Congratulations! You've won a brand new car, click here now.", "label": "spam"},
        {"text": "Hey, I received your email. I'll get back to you soon.", "label": "ham"}
    ]
    
    file_path = 'new_data.csv'
    
    # 파일이 이미 존재하는지 확인하여 헤더를 추가할지 결정
    if os.path.exists(file_path):
        df = pd.DataFrame(new_samples)
        # to_csv에 'quoting=csv.QUOTE_ALL' 옵션을 추가하여 텍스트에 쉼표가 있어도 문제가 없도록 함
        df.to_csv(file_path, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ {len(new_samples)}개의 새로운 데이터가 {file_path}에 추가되었습니다.")
    else:
        df = pd.DataFrame(new_samples)
        df.to_csv(file_path, mode='w', header=True, index=False)
        print(f"⚠️ {file_path} 파일이 없어 새로 생성하고 {len(new_samples)}개의 데이터를 추가했습니다.")

if __name__ == "__main__":
    generate_new_data()

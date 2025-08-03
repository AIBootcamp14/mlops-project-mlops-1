# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import os

def generate_and_save_new_data(output_dir='data/new_data', filename='new_spam_data.csv'):
    """
    새로운 스팸 및 햄 메일 데이터를 생성하고 CSV 파일로 저장하는 함수.
    
    이 스크립트는 실제 운영 환경에서 새로운 데이터가 수집되는 과정을
    간단하게 시뮬레이션하는 역할을 합니다.
    """
    print("새로운 데이터를 생성하고 있습니다...")

    # 새로운 스팸 메일 샘플
    new_spam_texts = [
        "Congratulations! You've won a brand new car. Click the link to claim your prize now: http://fakeurl.com/prize",
        "Free money just for you! We found a secret way to make cash fast. Visit http://fakeurl.com/cash",
        "You are a winner! Get your free gift card by texting YES to 12345.",
        "URGENT: Your account has been compromised. Log in immediately to fix the issue: http://fakeurl.com/security",
        "Limited time offer! Buy one get one free on all products. Don't miss out!"
    ]
    
    # 새로운 햄(정상) 메일 샘플
    new_ham_texts = [
        "Hey, are we still on for lunch tomorrow?",
        "Could you please review the attached document for the meeting?",
        "Here are the meeting notes from today's call.",
        "Just checking in to see how you're doing.",
        "I'll be a little late to work today, stuck in traffic."
    ]
    
    # 데이터프레임 생성
    # 스팸: target=1, 햄: target=0
    spam_df = pd.DataFrame({'text': new_spam_texts, 'target': 1})
    ham_df = pd.DataFrame({'text': new_ham_texts, 'target': 0})
    
    # 두 데이터프레임을 합치기
    new_data_df = pd.concat([spam_df, ham_df], ignore_index=True)
    
    # 데이터를 무작위로 섞기
    new_data_df = new_data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 저장할 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV 파일로 저장
    output_path = os.path.join(output_dir, filename)
    new_data_df.to_csv(output_path, index=False)
    
    print(f"새로운 데이터 {len(new_data_df)}개를 '{output_path}'에 성공적으로 저장했습니다.")

if __name__ == "__main__":
    generate_and_save_new_data()

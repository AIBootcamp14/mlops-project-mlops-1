# 필요한 라이브러리를 가져옵니다.
import pandas as pd
import os
import random

def generate_new_data(full_data_path, new_data_dir, chunk_size=100):
    """
    대규모 데이터셋에서 랜덤으로 데이터를 샘플링하여 새로운 데이터를 생성합니다.

    Args:
        full_data_path (str): 대규모 데이터셋 파일의 경로.
        new_data_dir (str): 새로운 데이터 파일을 저장할 디렉토리.
        chunk_size (int): 한 번에 샘플링할 데이터의 크기.
    """
    print("✨ 새로운 데이터를 생성합니다.")

    # 새로운 데이터를 저장할 디렉토리가 없으면 생성합니다.
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)

    try:
        # 대규모 데이터셋 파일을 읽어옵니다.
        df = pd.read_csv(full_data_path, encoding='latin-1', sep=',', quotechar='"', on_bad_lines='skip')
        print(f"✅ 대규모 데이터셋을 성공적으로 로드했습니다. 총 데이터 크기: {len(df)}")
    except FileNotFoundError:
        print(f"❌ 오류: '{full_data_path}' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return
    except Exception as e:
        print(f"❌ 오류: 데이터 파일을 읽는 중 문제가 발생했습니다: {e}")
        return

    # 'target'과 'text' 컬럼이 있는지 확인합니다.
    if 'target' not in df.columns or 'text' not in df.columns:
        print("❌ 오류: 데이터에 'target' 또는 'text' 컬럼이 없습니다.")
        return

    # 데이터가 충분한지 확인합니다.
    if len(df) < chunk_size:
        print(f"❌ 경고: 데이터가 너무 적어 {chunk_size}개의 샘플을 추출할 수 없습니다. 전체 데이터를 사용합니다.")
        sampled_df = df
    else:
        # 데이터셋에서 랜덤으로 'chunk_size'만큼 샘플링합니다.
        sampled_df = df.sample(n=chunk_size, random_state=random.randint(0, 10000))
        print(f"✅ 대규모 데이터셋에서 {chunk_size}개의 새로운 데이터를 샘플링했습니다.")

    # 샘플링된 데이터를 CSV 파일로 저장합니다.
    output_path = os.path.join(new_data_dir, 'new_spam_data.csv')
    sampled_df.to_csv(output_path, index=False)
    print(f"✅ 새로운 데이터가 '{output_path}'에 저장되었습니다.")


if __name__ == "__main__":
    # 대규모 데이터셋 경로와 새로운 데이터 저장 경로를 정의합니다.
    full_dataset_path = 'data/full_spam_dataset.csv'
    new_data_directory = 'data/new_data'
    generate_new_data(full_dataset_path, new_data_directory)

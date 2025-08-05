import pandas as pd
import pymysql

db_config = {
    'host': 'my-mlops-db',
    'user': 'root',
    'password': 'root',
    'database': 'mlops'
}

def save_csv_to_mysql(csv_path, db_config, table_name='hamspam'):
    # CSV 파일 로딩
    df = pd.read_csv(csv_path)

    # NaN 값을 빈 문자열로 대체
    df.fillna('', inplace=True)

    # MySQL 연결
    connection = pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
        charset='utf8mb4'
    )

    try:
        with connection.cursor() as cursor:
            ## 테이블 생성 (없을 경우)
            #create_table_sql = f"""
            #CREATE TABLE IF NOT EXISTS {table_name} (
            #    id INT AUTO_INCREMENT PRIMARY KEY,
            #    target INT,
            #    text TEXT,
            #    processed_text TEXT
            #);
            #"""
            #cursor.execute(create_table_sql)

            # 데이터 삽입
            for _, row in df.iterrows():
                insert_sql = f"""
                INSERT INTO {table_name} (target, text, processed_text)
                VALUES (%s, %s, %s);
                """
                cursor.execute(insert_sql, (row['target'], row['text'], row['processed_text']))

        # 커밋 후 연결 종료
        connection.commit()
        print(f"✅ {csv_path} 파일이 MySQL 테이블 '{table_name}'에 성공적으로 저장되었습니다!")

    except Exception as e:
        print("⚠️ 에러 발생:", e)
    finally:
        connection.close()

if __name__ == '__main__':
    print('table생성 시작!!')
    save_csv_to_mysql('./hamspam_processed.csv', db_config)
    print('table생성 종료!!')


# Docker 이미지의 베이스 이미지로 Python 3.10 slim 버전을 사용합니다.
# slim 버전은 더 작고 보안에 유리합니다.
FROM python:3.10-slim-buster

# 작업 디렉토리를 /app으로 설정합니다.
# 이 디렉토리 안에서 모든 애플리케이션 관련 작업이 이루어집니다.
WORKDIR /app

# 현재 디렉토리의 모든 파일 (requirements.txt 포함)을 /app 디렉토리로 복사합니다.
# .dockerignore 파일을 사용하여 불필요한 파일 복사를 방지할 수 있습니다.
COPY . /app

# Python 패키지 설치 시 캐시를 사용하지 않아 이미지 크기를 줄이고 빌드 일관성을 높입니다.
# --break-system-packages는 시스템 패키지 보호를 무시하고 설치를 강제합니다.
# NLTK 데이터도 자동으로 다운로드합니다.
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"

# FastAPI 애플리케이션이 실행될 포트를 노출합니다.
# 이 포트는 컨테이너 외부에서 접근 가능하도록 설정됩니다.
EXPOSE 8000

# 컨테이너가 시작될 때 실행될 명령어입니다.
# Uvicorn을 사용하여 FastAPI 애플리케이션을 실행합니다.
# main.py 파일이 api 폴더 안에 있으므로 'api.main:app'으로 경로를 지정합니다.
# --host 0.0.0.0: 모든 네트워크 인터페이스에서 요청을 수신합니다.
# --port 8000: 8000번 포트에서 실행합니다.
# --reload는 개발 환경에서만 사용하고, 프로덕션 환경에서는 제거하는 것이 좋습니다.
# 여기서는 개발 편의를 위해 포함합니다.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

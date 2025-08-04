# **스팸 이메일 분류기 MLOps 파이프라인 구축**  
이 프로젝트는 스팸 이메일 분류 모델을 개발하고, 이를 프로덕션 환경에 배포하기 위한 완전 자동화된 MLOps(Machine Learning Operations) 파이프라인을 구축하는 것을 목표로 합니다. 데이터 모니터링부터 모델 재학습, 배포까지 전 과정을 자동화하여, 모델 성능을 지속적으로 관리하고 서비스의 안정성을 보장합니다.

# test3

- **프로젝트 기간:** 2025.07.28 ~ 2025.08.08  
- **배포 링크:** 서비스 바로가기 (Docker Hub 링크)
---

## **1. 서비스 구성 요소**  
### **1.1 주요 기능**  
- 데이터 드리프트 모니터링: 데이터의 통계적 변화를 주기적으로 감지하여 모델 재학습 필요 여부 판단

- 모델 재학습 자동화: 신규 데이터가 감지되면 GitHub Actions를 통해 모델 재학습 파이프라인 자동 트리거

- MLflow 기반 실험 추적: 모델 성능 지표(accuracy, f1-score 등) 및 파라미터를 자동 기록 및 비교

- MLflow 모델 레지스트리: 학습된 모델을 버전별로 관리하고, 배포할 모델을 체계적으로 선택

- 조건부 배포(Conditional Deployment): 신규 모델의 성능이 이전 모델보다 우수할 때만 Docker 이미지로 빌드 및 배포

- 컨테이너화: 모델을 Docker 이미지로 패키징하여 재현 가능한 배포 환경 구축

### **1.2 파이프라인 사용자 흐름**  
- 이 프로젝트는 사람의 개입 없이 아래와 같은 MLOps 파이프라인이 자동 실행됩니다.

1. 데이터 수집: 대규모 데이터셋에서 정기적으로 새로운 데이터 샘플을 가져와 기존 데이터에 추가합니다.

2. 모니터링: model_monitor.py가 데이터의 통계적 변화를 감지하고, 재학습이 필요하면 retrain_needed 변수를 true로 설정합니다.

3. CI(지속적 통합): 재학습이 필요한 경우, train_model.py를 실행하여 새로운 데이터로 모델을 재학습하고, MLflow에 기록합니다.

4. 성능 비교: 재학습된 모델의 성능을 MLflow에 기록된 이전 최고 성능 모델과 비교합니다.

5. CD(지속적 배포): 신규 모델의 성능이 더 좋으면 deploy 잡이 실행되어 Docker 이미지를 Docker Hub에 푸시합니다.
---

## **2. 활용 장비 및 협업 툴**  

### **2.1 활용 장비**  
- **실행 환경:** *GitHub Actions (ubuntu-latest)*  
- **개발 환경:** *Python 3.10*  
- **컨테이너 환경:** *Docker*  

### **2.2 협업 툴**  
- **소스 관리:** GitHub  
- **프로젝트 관리:** Notion  
- **커뮤니케이션:** Slack  
- **버전 관리:** Git  

---

## **3. 최종 선정 AI 모델 구조**  
- **모델 이름:** *Multinomial Naive Bayes*  
- **구조 및 설명:** *텍스트 분류에 효과적인 확률 기반의 모델입니다. 대용량 텍스트 데이터를 효율적으로 처리하며, TfidfVectorizer를 사용하여 텍스트 데이터를 벡터화한 후 학습을 진행합니다.*  
- **학습 데이터:** *Kaggle의 Spam Text Message Classification 데이터셋을 활용하여 시뮬레이션했습니다.*  
- **평가 지표:** *모델의 예측 성능을 평가하기 위해 다음 지표들을 사용했습니다.*  
정확도(Accuracy): 전체 예측 중 올바르게 예측한 비율.

정밀도(Precision): 스팸으로 예측한 것 중 실제 스팸의 비율.

재현율(Recall): 실제 스팸 중 스팸으로 올바르게 예측한 비율.

F1-Score: 정밀도와 재현율의 조화 평균.

---

## **4. 서비스 아키텍처**  
### **4.1 시스템 구조도**  
아래는 프로젝트의 전체적인 파이프라인을 나타내는 구조도입니다. 

![MLOps 파이프라인 구조도](docs/images/architecture.svg)

### **4.2 데이터 흐름도**  
1. 데이터 수집: src/data_ingestion.py가 대규모 데이터셋(data/full_spam_dataset.csv)에서 새로운 데이터 100개를 샘플링하여 data/new_data/new_spam_data.csv에 저장합니다.

2. 데이터 처리: notebooks/train_model.py가 원본 데이터와 새로 수집된 데이터를 합친 후, 텍스트 전처리 및 TF-IDF 벡터화를 수행합니다.

3. 모델 학습: 처리된 데이터를 바탕으로 Multinomial Naive Bayes 모델을 학습합니다.

4. 모델 레지스트리: 학습된 모델의 성능이 기존 최고 모델보다 좋으면 MLflow 모델 레지스트리에 새로운 버전으로 등록합니다.

5. 배포: 성능이 개선된 모델은 Docker 이미지로 빌드되어 Docker Hub에 푸시됩니다. 

---

## **5. 사용 기술 스택**  
### **5.1 백엔드**  
- Flask / FastAPI / Django *(필요한 항목 작성)*  
- 데이터베이스: SQLite / PostgreSQL / MySQL  

### **5.2 프론트엔드**  
- React.js / Next.js / Vue.js *(필요한 항목 작성)*  

### **5.3 머신러닝 및 데이터 분석**  
MLflow: 실험 추적 및 모델 레지스트리

scikit-learn: 모델 학습 및 평가

Pandas: 데이터 처리 및 분석

NLTK: 텍스트 전처리 (어간 추출 등)

### **5.4 배포 및 운영**  
GitHub Actions: CI/CD 파이프라인 자동화

Docker: 모델 컨테이너화 및 배포

Python: 백엔드 스크립트 및 모델 개발 

---

## **6. 팀원 소개**  

| 이름      | 역할              | GitHub                               | 담당 기능                                 |
|----------|------------------|-------------------------------------|-----------------------------------------|
| **김시진** | 팀장 | [GitHub 링크](링크 입력)             | 서버 구축, API 개발, 배포 관리            |
| **김영** | 미정  | [GitHub 링크](링크 입력)             | UI/UX 디자인, 프론트엔드 개발             |
| **정예은** | 미정    | [GitHub 링크](링크 입력)             | AI 모델 선정 및 학습, 데이터 분석         |
| **전수정** | 미정    | [GitHub 링크](링크 입력)             | 데이터 수집, 전처리, 성능 평가 및 테스트   |

---

## **7. Appendix**  
### **7.1 참고 자료**  
- 데이터 출처: Kaggle - Spam Text Message Classification

- MLOps 개념: MLflow Documentation 

### **7.2 설치 및 실행 방법**  
1. **프로젝트 복제(Clone):**  
    ```bash
    git clone https://github.com/AIBootcamp14/mlops-project-mlops-1.git

    cd mlops-project-mlops-1
    ```

2. **MLflow UI 실행 (로컬 환경에서 확인):**  
    ```bash
    # 먼저 파이프라인을 한 번 실행해서 mlruns 폴더를 생성해야 합니다.
    # GitHub Actions에서 다운로드한 mlflow-artifacts 압축 해제 후, 아래 명령어를 실행하세요.
    python -m mlflow ui --backend-store-uri file:///your/local/path/to/mlflow-artifacts

    ```

3. **웹페이지 접속:**  
    ```
    http://127.0.0.1:5000
    ```




#!/bin/bash
# scripts/run_preprocessing.sh
# 간단한 데이터 전처리 자동화 스크립트

set -e  # 오류 발생시 스크립트 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              스팸메일 데이터 전처리 자동화                         ║"
echo "║                   (노트북 버전과 동일)                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 함수 정의
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 환경 확인
log_info "Python 환경 확인 중..."
if ! command -v python3 &> /dev/null; then
    log_error "Python3가 설치되지 않았습니다."
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
log_info "Python 버전: $python_version"

# 2. 필요한 디렉토리 생성
log_info "디렉토리 구조 생성 중..."
mkdir -p data
mkdir -p data/processed
mkdir -p logs
mkdir -p models

# 3. 가상환경 설정 (선택적)
if [ ! -d "venv" ]; then
    log_info "가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
if [ -d "venv" ]; then
    log_info "가상환경 활성화 중..."
    source venv/bin/activate
fi

# 4. 필요한 패키지 설치
log_info "필요한 패키지 설치 중..."
pip install -q pandas scikit-learn nltk

# NLTK 데이터 다운로드
python3 -c "
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print('NLTK 데이터 이미 존재')
except LookupError:
    print('NLTK 데이터 다운로드 중...')
    nltk.download('punkt', quiet=True)
"

# 5. 데이터 파일 확인
if [ ! -f "data/spam.csv" ]; then
    log_warn "data/spam.csv 파일이 없습니다."
    log_info "샘플 데이터를 생성합니다..."
    
    # 간단한 샘플 데이터 생성
    cat > data/spam.csv << 'EOF'
target,text
ham,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
spam,"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
ham,"U dun say so early hor... U c already then say..."
spam,"FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"
ham,"Even my brother is not like to speak with me. They treat me like aids patent."
spam,"WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
ham,"As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"
spam,"Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"
ham,"I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."
spam,"SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"
EOF
    
    log_info "✅ 샘플 데이터 생성 완료"
else
    log_info "✅ 데이터 파일 확인됨: data/raw/spam.csv"
fi

# 6. 전처리 스크립트 실행
log_info "데이터 전처리 실행 중..."
python3 data/scripts/preprocess_data.py

# 7. 결과 확인
if [ -f "data/processed/train_data_latest.csv" ] && [ -f "data/processed/test_data_latest.csv" ]; then
    log_info "✅ 전처리 완료!"
    
    echo -e "${GREEN}"
    echo "📊 결과 파일:"
    echo "  - data/processed/train_data_latest.csv"
    echo "  - data/processed/test_data_latest.csv"
    echo "  - data/processed/processed_data_latest.csv"
    echo "  - data/processed/preprocessing_report.json"
    echo -e "${NC}"
    
    # 간단한 통계 출력
    log_info "데이터 통계:"
    python3 -c "
import pandas as pd
import os

try:
    if os.path.exists('data/processed/train_data_latest.csv'):
        train_df = pd.read_csv('data/processed/train_data_latest.csv')
        test_df = pd.read_csv('data/processed/test_data_latest.csv')
        
        print(f'📈 훈련 데이터: {len(train_df):,}개 샘플')
        print(f'📈 테스트 데이터: {len(test_df):,}개 샘플')
        print(f'📈 총 데이터: {len(train_df) + len(test_df):,}개 샘플')
        
        print('\\n📊 훈련 데이터 타겟 분포:')
        target_counts = train_df['target'].value_counts()
        for target, count in target_counts.items():
            label = 'ham' if target == 0 else 'spam'
            print(f'   {label}: {count:,}개 ({count/len(train_df)*100:.1f}%)')
        
        print('\\n🔤 전처리된 텍스트 예시:')
        for i in range(min(3, len(train_df))):
            text = train_df.iloc[i]['text'][:50]
            target = 'ham' if train_df.iloc[i]['target'] == 0 else 'spam'
            print(f'   [{target}] {text}...')
            
except Exception as e:
    print(f'통계 출력 실패: {e}')
"
    
    echo -e "${BLUE}"
    echo "🎯 다음 단계:"
    echo "  1. 모델 훈련: python3 src/models/train.py"
    echo "  2. API 실행: python3 api/main.py"
    echo "  3. 테스트: python3 -m pytest tests/"
    echo -e "${NC}"
    
else
    log_error "❌ 전처리 실패! 결과 파일이 생성되지 않았습니다."
    
    if [ -f "logs/preprocessing.log" ]; then
        log_info "로그 파일 확인:"
        echo "$(tail -10 logs/preprocessing.log)"
    fi
    
    exit 1
fi

echo -e "${GREEN}"
echo "🎉 스팸메일 데이터 전처리 자동화 완료!"
echo -e "${NC}"
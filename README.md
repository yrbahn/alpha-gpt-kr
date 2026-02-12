# 🥧 Alpha-GPT-KR

**한국 증시를 위한 LLM 기반 자동 알파 마이닝 시스템**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Alpha-GPT 논문을 한국 증시에 맞게 구현한 프로젝트입니다. LLM(GPT-4)과 Genetic Programming을 사용하여 자동으로 최적의 알파 팩터를 생성하고, 실제 매매에 적용할 수 있습니다.

---

## 📋 목차

- [특징](#-특징)
- [성과](#-성과)
- [아키텍처](#-아키텍처)
- [설치](#-설치)
- [사용법](#-사용법)
- [실전 매매](#-실전-매매)
- [대시보드](#-대시보드)
- [API 설정](#-api-설정)
- [참고 논문](#-참고-논문)
- [라이선스](#-라이선스)

---

## ✨ 특징

### 🤖 LLM 기반 알파 생성
- **GPT-4**가 투자 아이디어를 분석하고 Python 코드로 알파 표현식 직접 생성
- 복잡한 팩터 조합 자동화
- 10개 알파 후보 생성 후 백테스트 평가

### 🧬 Genetic Programming 진화
- 30세대 진화로 알파 최적화
- 교차(70%) + 변이(30%) 연산
- IC (Information Coefficient) 기반 적합도 평가
- **IC 0.4773 달성** (단순 모멘텀 대비 100배 개선!)

### 💾 PostgreSQL 통합
- 한국 증시 데이터 (marketsense DB)
- 시가총액 상위 500개 종목 지원
- 2년 이상 백테스트 가능

### 📊 실시간 대시보드
- HTML 기반 웹 대시보드
- 5분마다 자동 업데이트
- 계좌 현황, 포트폴리오, 알파 스코어 시각화
- Chart.js로 차트 렌더링

### 🔥 실전 매매 지원
- **한국투자증권(KIS) API** 통합
- DB 기반 워크플로우: 알파 계산 → DB 저장 → 다음날 매수
- Stop-loss / Take-profit 리스크 관리
- 토큰 캐싱으로 API 제한 회피

---

## 🏆 성과

### Alpha-GPT가 생성한 최상위 알파

```python
AlphaOperators.ts_rank(
    AlphaOperators.ts_mean(returns, 2), 
    10
)
```

**성능:**
- **IC: 0.4773** (Information Coefficient)
- **백테스트 기간**: 2024-02-01 ~ 2026-02-12
- **종목 수**: 100개 (시가총액 상위)

**해석:**  
2일 평균 수익률의 10일 순위를 계산. 단기 모멘텀이 강한 종목을 선택.

### 성과 비교

#### 1일 Forward (초단기 전략)

| 방법 | 알파 | IC |
|------|------|-----|
| 간단한 모멘텀 | `ts_delta(close, 26)` | 0.0045 |
| LLM 생성 | `ts_rank(ts_std(returns,10)/ts_std(returns,20), 10)` | 0.0467 |
| **LLM + GP 진화** | `ts_rank(ts_mean(returns, 2), 10)` | **0.4773** |

**IC 개선:** 0.0045 → 0.4773 = **106배 증가!**

**거래비용 분석:**
- 연간 리밸런싱: ~250회
- 연간 거래비용: ~150% (0.3% × 2 × 250회)
- 높은 IC이지만 거래비용 부담 큼

#### 15일 Forward (논문 표준 - 월 2회 리밸런싱)

| 방법 | 알파 | IC | 연간 거래비용 |
|------|------|-----|---------------|
| LLM + GP (500 stocks) | GPT-4o 생성 알파 | **0.0311** | **~14.4%** |

**특징:**
- 논문 표준 방식 (월 2회 리밸런싱)
- 연간 리밸런싱: ~24회
- 연간 거래비용: ~14.4% (0.3% × 2 × 24회)
- **순수익**: 1일 전략 대비 유리할 가능성 (낮은 거래비용)
- 500개 종목 = 메모리 안정성 한계

**권장 전략:**
- 소액 투자: 15일 forward (거래비용 최소화)
- 대량 투자: 1일 forward (높은 IC 활용)

---

## 🏗️ 아키텍처

### 논문 방식 (3단계)

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Ideation (아이디어 정제)               │
│   - LLM이 투자 아이디어 분석                                │
│   - 필요한 데이터 필드 식별                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              2. Implementation (알파 생성 + 진화)            │
│   - LLM이 10개 알파 표현식 생성                             │
│   - GP가 30세대 진화 (교차, 변이, 선택)                     │
│   - IC 기반 적합도 평가                                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  3. Review (백테스트 + 선택)                 │
│   - 상위 5개 알파 백테스트                                   │
│   - IC, Sharpe, MDD 계산                                    │
│   - 최상위 알파 선택 → DB 저장                              │
└─────────────────────────────────────────────────────────────┘
```

### 실전 매매 워크플로우

```
[오후 5시] 알파 계산 → DB 저장
     ↓
[다음날 오전 9시] DB에서 읽기 → 매수 실행
     ↓
[언제든지] 대시보드로 현황 확인
```

---

## 🚀 설치

### 1. 저장소 클론

```bash
git clone https://github.com/yrbahn/alpha-gpt-kr.git
cd alpha-gpt-kr
```

### 2. Python 환경

```bash
# Python 3.9 이상 필요
python3 --version

# 가상환경 생성 (선택사항)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

**주요 패키지:**
- `openai`: GPT-4 API
- `psycopg2-binary`: PostgreSQL 연결
- `pandas`, `numpy`: 데이터 처리
- `loguru`: 로깅

### 4. 환경 변수 설정

`.env` 파일 생성:

```bash
cp .env.example .env
```

`.env` 파일 편집:

```env
# OpenAI API
OPENAI_API_KEY=sk-...

# PostgreSQL (marketsense DB)
DB_HOST=192.168.0.248
DB_PORT=5432
DB_NAME=marketsense
DB_USER=yrbahn
DB_PASSWORD=1234

# 한국투자증권 API (선택사항 - 실전 매매용)
KIS_APP_KEY=...
KIS_APP_SECRET=...
KIS_ACCOUNT_NO=...
KIS_MODE=real  # 또는 virtual
```

### 5. DB 스키마 생성

```bash
python3 setup_db.py
```

생성되는 테이블:
- `alpha_scores`: 알파 점수 저장
- `trading_signals`: 매매 신호
- `trading_portfolio`: 포트폴리오 이력
- `trading_account`: 계좌 상태
- `alpha_performance`: 알파 성과 추적

---

## 📖 사용법

### 방법 1: Alpha-GPT 완전판 (LLM + GP)

**가장 강력한 방법 - 논문 방식 그대로**

```bash
python3 alpha_gpt_with_gp.py
```

**프로세스:**
1. GPT-4가 5개 초기 알파 생성
2. GP가 30세대 진화 (10-15분 소요)
3. 최상위 알파 DB 저장

**결과:**
```
🏆 최상위 알파
IC: 0.4773
공식: AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 2), 10)
```

### 방법 2: LLM만 사용 (빠른 테스트)

```bash
python3 simple_alpha_gpt.py
```

**프로세스:**
1. GPT-4가 10개 알파 생성
2. 각 알파 백테스트 (2-3분 소요)
3. 상위 5개 선택

### 방법 3: 간단한 알파 적용

500개 종목에 특정 알파 적용:

```bash
python3 calculate_alpha_top500.py
```

---

## 💰 실전 매매

### 1. 알파 계산 및 DB 저장 (매일 저녁)

```bash
python3 calculate_and_save_alpha.py
```

- 시가총액 상위 500개 종목 분석
- 알파 계산
- DB에 저장 (alpha_scores 테이블)

### 2. 매수 실행 (다음날 아침)

#### 방법 2-1: 시총 상위 1000개에서 선택 (권장)

**시뮬레이션:**
```bash
python3 trade_top1000.py --top-n 8 --amount 5000000 --dry-run
```

**실제 매수:**
```bash
python3 trade_top1000.py --top-n 8 --amount 5000000
```

**특정 종목 제외:**
```bash
python3 trade_top1000.py --top-n 8 --amount 5000000 --exclude 042700 005690
```

**옵션:**
- `--top-n 8`: 상위 8개 종목
- `--amount 5000000`: 총 투자금 500만원
- `--exclude 042700 005690`: 제외할 종목 코드
- `--dry-run`: 시뮬레이션 모드

**장점:**
- 시가총액 상위 1000개 종목으로 범위 확대
- 거래정지 종목 자동 필터링
- 더 넓은 선택지로 과적합 방지

#### 방법 2-2: 기본 매매 (시총 500개)

**시뮬레이션:**
```bash
python3 trade_from_db.py --top-n 15 --amount 5000000 --dry-run
```

**실제 매수:**
```bash
python3 trade_from_db.py --top-n 15 --amount 5000000
```

**옵션:**
- `--top-n 15`: 상위 15개 종목
- `--amount 5000000`: 총 투자금 500만원
- `--dry-run`: 시뮬레이션 모드

### 3. 리스크 관리

- **Stop-loss**: -5% 손실 시 자동 청산
- **Take-profit**: +10% 수익 시 자동 청산
- **리밸런싱**: 5일마다 재조정 권장

---

## 📊 대시보드

### 서버 시작

```bash
./start_dashboard.sh
```

**접속:**
- **로컬**: http://localhost:9999/dashboard.html
- **외부**: http://YOUR_IP:9999/dashboard.html

### 기능

- 📈 **계좌 현황**: 총 자산, 현금, 수익률
- 📊 **포트폴리오**: 종목별 손익, 비중
- 🎯 **알파 스코어**: 상위 종목 리스트
- 📉 **차트**: 계좌 가치 추이, 알파 분포
- 🔔 **매매 신호**: 최근 매수/매도 신호

### 자동 업데이트

- 서버: 5분마다 DB 조회 → 대시보드 재생성
- 브라우저: 5분마다 자동 새로고침

### 백그라운드 실행

```bash
nohup ./start_dashboard.sh > logs/dashboard.log 2>&1 &
```

---

## 🔑 API 설정

### OpenAI API

1. https://platform.openai.com 에서 API 키 발급
2. `.env`에 `OPENAI_API_KEY` 설정
3. GPT-4 권한 필요

**비용:**
- Alpha-GPT 1회 실행: 약 $0.50~1.00
- LLM 호출: 10-20회
- 토큰 사용량: 5,000~10,000 tokens

### 한국투자증권 API (선택사항)

1. https://apiportal.koreainvestment.com 에서 앱 등록
2. APP Key, APP Secret 발급
3. `.env`에 설정

**참고:**
- 토큰 유효기간: 24시간
- 토큰 생성 제한: 1분당 1회
- 토큰 캐싱 구현: `~/.kis_tokens/`

---

## 📁 프로젝트 구조

```
alpha-gpt-kr/
├── alpha_gpt_kr/              # 메인 패키지
│   ├── agents/                # LLM 에이전트
│   │   ├── trading_idea_polisher.py
│   │   ├── quant_developer.py
│   │   └── analyst.py
│   ├── mining/                # 알파 마이닝
│   │   ├── operators.py       # 알파 연산자
│   │   └── genetic_programming.py
│   ├── data/                  # 데이터 로더
│   │   └── postgres_loader.py
│   ├── backtest/              # 백테스트 엔진
│   │   └── engine.py
│   ├── trading/               # 실전 매매
│   │   ├── kis_api.py
│   │   └── trader.py
│   └── core.py                # AlphaGPT 메인 클래스
│
├── alpha_gpt_with_gp.py       # ⭐ LLM + GP 진화
├── simple_alpha_gpt.py        # LLM만 사용
├── calculate_alpha_top500.py  # 500개 종목 알파 계산
├── trade_from_db.py           # DB 기반 매수
├── generate_dashboard.py      # 대시보드 생성
├── start_dashboard.sh         # 대시보드 서버
├── setup_db.py                # DB 초기화
│
├── db_schema.sql              # DB 스키마
├── requirements.txt           # 의존성
├── .env                       # 환경 변수 (git ignore)
├── .gitignore
└── README.md
```

---

## 🧪 테스트

### 단위 테스트

```bash
# PostgreSQL 연결 테스트
python3 test_postgres.py

# KIS API 테스트
python3 test_kis_trading.py

# 알파 계산 테스트
python3 test_alphagpt_postgres.py

# GP 진화 테스트
python3 test_gp_evolution.py
```

### 백테스트

```bash
# 2년 장기 백테스트
python3 test_longterm.py

# LLM 알파 백테스트
python3 test_llm_longterm.py
```

---

## 📈 성능 최적화

### 데이터 로드 최적화

```python
# 전체 데이터 (느림)
data = loader.load_data()

# 기간 제한 (빠름)
data = loader.load_data(
    start_date="2025-11-01",
    end_date="2026-02-12"
)

# 종목 제한
data = loader.load_data(universe=top_100_tickers)
```

### GP 진화 최적화

```python
# 빠른 테스트
genetic_programming_evolution(
    seed_alphas=seeds,
    data=data,
    generations=10,       # 10세대 (2-3분)
    population_size=10    # 개체수 10
)

# 완전한 진화
genetic_programming_evolution(
    seed_alphas=seeds,
    data=data,
    generations=30,       # 30세대 (10-15분)
    population_size=20    # 개체수 20
)
```

---

## 🛡️ 보안 및 주의사항

### 환경 변수 보호

⚠️ **절대 `.env` 파일을 Git에 커밋하지 마세요!**

```bash
# .env를 실수로 커밋한 경우
git rm --cached .env
git commit -m "Remove .env from tracking"
```

### API 키 보호

- OpenAI API 키는 절대 코드에 하드코딩하지 마세요
- 환경 변수 또는 시크릿 관리자 사용
- GitHub Actions 사용 시 Secrets 활용

### 실전 매매 주의

⚠️ **실제 돈을 투자하기 전에:**

1. **가상계좌로 먼저 테스트** (`KIS_MODE=virtual`)
2. **백테스트 결과 검증** (최소 1년 이상)
3. **리스크 관리 설정** (Stop-loss, Take-profit)
4. **소액으로 시작** (전체 자산의 5% 이내)
5. **지속적 모니터링** (대시보드 활용)

---

## 🐛 문제 해결

### PostgreSQL 연결 실패

```bash
# 연결 테스트
psql -h 192.168.0.248 -U yrbahn -d marketsense

# 방화벽 확인
telnet 192.168.0.248 5432
```

### OpenAI API 오류

```python
# 에러: "Rate limit exceeded"
# 해결: 요청 간격 늘리기
import time
time.sleep(1)  # API 호출 사이에 1초 대기
```

### KIS API 토큰 에러

```bash
# 토큰 캐시 삭제
rm -rf ~/.kis_tokens/

# 재시도
python3 check_balance.py
```

### 대시보드 접속 안 됨

```bash
# macOS 방화벽 확인
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Python 허용
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/python3
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/python3
```

---

## 📚 참고 논문

**Alpha-GPT: An Agent for Stock Alpha Mining**

- arXiv: [2308.00016](https://arxiv.org/abs/2308.00016)
- Authors: Xiao Gao, et al.
- Published: 2023

**핵심 개념:**
- LLM을 사용한 자동 알파 생성
- Genetic Programming 기반 최적화
- IC (Information Coefficient) 평가

---

## 🤝 기여

Pull Request를 환영합니다!

### 기여 가이드

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 코딩 스타일

- PEP 8 준수
- Type hints 사용
- Docstrings 작성

---

## 📝 TODO

- [ ] 다중 LLM 지원 (Claude, Gemini)
- [ ] 더 많은 알파 연산자 추가
- [ ] 포트폴리오 최적화 (Mean-Variance)
- [ ] 실시간 모니터링 알림 (Telegram, Discord)
- [ ] 웹 UI 개선 (React)
- [ ] 클라우드 배포 가이드 (AWS, GCP)

---

## 📄 라이선스

MIT License

Copyright (c) 2026 Youngrok Bahn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📞 문의

- **GitHub Issues**: https://github.com/yrbahn/alpha-gpt-kr/issues
- **Email**: yrbahn@example.com
- **Twitter**: @yrbahn

---

## 🙏 감사

- **Alpha-GPT 논문 저자들**
- **OpenAI** (GPT-4 API)
- **한국투자증권** (KIS API)
- **PostgreSQL 커뮤니티**

---

**⚠️ 면책 조항**

이 프로젝트는 교육 및 연구 목적으로 제공됩니다. 실제 투자에 사용할 경우 발생하는 손실에 대해 저자는 책임지지 않습니다. 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.

**투자 원칙:**
- 과거 성과가 미래 수익을 보장하지 않습니다
- 잃어도 괜찮은 금액만 투자하세요
- 분산 투자하세요
- 지속적으로 학습하고 개선하세요

---

**Made with 🥧 in Korea**

# 한국투자증권 실전 매매 시스템 설정 가이드

## 🔑 1단계: API Key 발급

### 한국투자증권 API 신청
1. **한국투자증권 계좌 개설** (있으면 skip)
2. **API 서비스 신청**
   - 웹사이트: https://apiportal.koreainvestment.com
   - 회원가입 후 로그인
   - "App 생성" 클릭
   - **모의투자** 또는 **실전투자** 선택

3. **App Key 발급**
   - App Key (앱 키)
   - App Secret (앱 시크릿)
   - 계좌번호 (8자리-2자리, 예: 12345678-01)

---

## ⚙️ 2단계: 환경변수 설정

`.env` 파일에 다음 정보 추가:

```bash
# 한국투자증권 KIS API
KIS_APP_KEY=your_app_key_here
KIS_APP_SECRET=your_app_secret_here
KIS_ACCOUNT_NO=12345678-01
```

**보안 주의**: `.env` 파일을 절대 GitHub에 올리지 마세요!

---

## 🧪 3단계: 모의투자로 테스트

```bash
# 테스트 실행
python test_kis_trading.py

# 선택 메뉴:
# 1. KIS API 기본 테스트 (잔고, 보유종목, 현재가 조회)
# 2. Alpha-GPT 자동 매매 테스트
# 3. 전체 테스트
```

### 주요 테스트 항목
- ✅ 계좌 잔고 조회
- ✅ 보유 종목 조회
- ✅ 현재가 조회
- ✅ 알파 신호 생성
- ✅ 포트폴리오 리밸런싱 (주석 처리됨)

---

## 🚀 4단계: 실전 투자 전환

**모의투자에서 충분히 검증 후 실전 전환!**

### `.env` 파일 수정:
```python
# trader.py 또는 test 스크립트에서:
api = KISApi(
    app_key=os.getenv("KIS_APP_KEY"),
    app_secret=os.getenv("KIS_APP_SECRET"),
    account_no=os.getenv("KIS_ACCOUNT_NO"),
    is_real=True  # ← False에서 True로 변경
)
```

---

## 📊 자동 매매 로직

### AlphaTrader 주요 기능

1. **알파 신호 생성**
   - Alpha-GPT가 생성한 최고 알파 팩터 사용
   - 상위 N개 종목 선택 (default: 10개)

2. **포트폴리오 리밸런싱**
   - 주기: 5영업일 (변경 가능)
   - 동일 비중 매수
   - 신호 없는 종목 매도

3. **리스크 관리**
   - 손절매: -5% (변경 가능)
   - 익절: +10% (변경 가능)

### 사용 예시:
```python
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.trading.trader import AlphaTrader

# KIS API 초기화
api = KISApi(
    app_key="your_key",
    app_secret="your_secret",
    account_no="12345678-01",
    is_real=False  # 모의투자
)

# AlphaTrader 초기화
trader = AlphaTrader(
    kis_api=api,
    alpha_gpt=alpha_gpt_instance,
    max_stocks=10,
    rebalance_days=5,
    stop_loss_pct=-0.05,
    take_profit_pct=0.10
)

# 일일 체크 (손절/익절만)
trader.run_daily_check()

# 리밸런싱
trader.rebalance_portfolio(force=True)
```

---

## 🔒 보안 주의사항

1. **API Key 보안**
   - `.env` 파일을 `.gitignore`에 추가
   - 절대 GitHub에 업로드 금지

2. **모의투자 먼저**
   - 최소 1-2주 모의투자로 검증
   - 실전은 소액으로 시작

3. **리스크 관리**
   - 손절매/익절 설정 필수
   - 최대 보유 종목 수 제한
   - 투자금액 제한 설정

---

## 📅 자동화 (cron)

매일 자동 실행하려면 cron 설정:

```bash
# 매일 오전 9시: 리밸런싱
0 9 * * 1-5 cd /path/to/alpha-gpt-kr && python -c "from test_kis_trading import *; trader.rebalance_portfolio()"

# 매일 오후 3시: 손절/익절 체크
0 15 * * 1-5 cd /path/to/alpha-gpt-kr && python -c "from test_kis_trading import *; trader.run_daily_check()"
```

---

## 🆘 문제 해결

### API 오류
- **401 Unauthorized**: App Key/Secret 확인
- **토큰 만료**: 자동 재발급됨
- **주문 실패**: 계좌번호, 잔고 확인

### 데이터 문제
- PostgreSQL 연결 확인
- 최신 데이터 업데이트 여부

---

## 📞 지원

- 한국투자증권 API 문서: https://apiportal.koreainvestment.com
- OpenAPI Discord: https://discord.gg/KIS_API

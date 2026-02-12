# 🚀 Alpha-GPT-KR 실전 매매 시스템

한국투자증권 KIS API를 사용한 Alpha-GPT 기반 자동 매매 시스템입니다.

## ✨ 주요 기능

### 1. 알파 팩터 기반 매매
- ✅ Alpha-GPT가 생성한 최고 성능 알파 팩터 사용
- ✅ 상위 N개 종목 자동 선택
- ✅ 동일 비중 포트폴리오

### 2. 자동 리밸런싱
- ✅ 주기적 포트폴리오 재조정 (default: 5영업일)
- ✅ 신호 없는 종목 자동 매도
- ✅ 새로운 신호 종목 매수

### 3. 리스크 관리
- ✅ **손절매**: -5% (설정 가능)
- ✅ **익절**: +10% (설정 가능)
- ✅ 일일 체크 및 자동 매도

### 4. 모의투자 지원
- ✅ 실전 전환 전 모의투자로 검증
- ✅ 동일한 API, 안전한 테스트

---

## 📦 구조

```
alpha_gpt_kr/
└── trading/
    ├── __init__.py
    ├── kis_api.py          # 한국투자증권 API 클라이언트
    └── trader.py           # Alpha-GPT 자동 매매 시스템

test_kis_trading.py         # 테스트 스크립트
TRADING_SETUP.md           # 설정 가이드
```

---

## 🚀 빠른 시작

### 1. API Key 발급
1. https://apiportal.koreainvestment.com 접속
2. 회원가입 → App 생성
3. **모의투자** 또는 **실전투자** 선택
4. App Key, App Secret, 계좌번호 확인

### 2. 환경변수 설정
`.env` 파일에 추가:
```bash
KIS_APP_KEY=your_app_key_here
KIS_APP_SECRET=your_app_secret_here
KIS_ACCOUNT_NO=12345678-01
```

### 3. 테스트 실행
```bash
python test_kis_trading.py
```

---

## 💡 사용 예시

### 기본 사용
```python
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.trading.trader import AlphaTrader

# KIS API 초기화 (모의투자)
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

# 일일 체크 (손절/익절)
trader.run_daily_check()

# 리밸런싱
trader.rebalance_portfolio(force=True)
```

### KIS API 직접 사용
```python
# 계좌 정보
balance = api.get_balance()
holdings = api.get_holdings()

# 현재가 조회
price = api.get_current_price("005930")  # 삼성전자

# 주문
api.buy_stock("005930", 10)  # 10주 매수
api.sell_stock("005930", 5)  # 5주 매도
```

---

## 🎯 전략 파라미터

### AlphaTrader 설정
```python
trader = AlphaTrader(
    kis_api=api,
    alpha_gpt=alpha_gpt,
    
    # 포트폴리오 설정
    max_stocks=10,           # 최대 보유 종목 수
    rebalance_days=5,        # 리밸런싱 주기 (영업일)
    
    # 리스크 관리
    stop_loss_pct=-0.05,     # 손절매 -5%
    take_profit_pct=0.10     # 익절 +10%
)
```

---

## ⚠️ 주의사항

### 보안
1. **API Key 보안**
   - `.env` 파일을 절대 GitHub에 올리지 마세요
   - `.gitignore`에 `.env` 추가 필수

2. **모의투자 먼저**
   - 최소 1-2주 모의투자로 검증
   - 실전은 소액으로 시작

### 리스크 관리
1. **손절매/익절 설정 필수**
2. **최대 투자금액 제한**
3. **정기적인 모니터링**

---

## 📊 백테스트 vs 실전

| 항목 | 백테스트 | 실전 |
|------|---------|------|
| 슬리피지 | ❌ 없음 | ✅ 있음 |
| 거래비용 | ❌ 없음 | ✅ 있음 |
| 유동성 | ❌ 무제한 | ✅ 제한 |
| 심리적 요인 | ❌ 없음 | ✅ 큼 |

**실전 수익률 ≠ 백테스트 수익률**

---

## 🔄 자동화 (선택)

### cron 설정
```bash
# 매일 오전 9시: 리밸런싱
0 9 * * 1-5 cd /path/to/alpha-gpt-kr && python -c "from test_kis_trading import *; trader.rebalance_portfolio()"

# 매일 오후 3시: 손절/익절 체크
0 15 * * 1-5 cd /path/to/alpha-gpt-kr && python -c "from test_kis_trading import *; trader.run_daily_check()"
```

---

## 📞 지원

- **한국투자증권 API 문서**: https://apiportal.koreainvestment.com
- **API 문의**: KIS OpenAPI 고객센터

---

## 📝 변경 이력

### v1.0.0 (2026-02-12)
- ✅ 한국투자증권 KIS API 연동
- ✅ Alpha-GPT 기반 자동 매매
- ✅ 리스크 관리 (손절/익절)
- ✅ 모의투자 지원

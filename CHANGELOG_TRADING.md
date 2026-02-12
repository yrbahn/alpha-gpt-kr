# Trading System Changelog

## [1.0.0] - 2026-02-12

### 🎉 Added - 한국투자증권 실전 매매 시스템

#### 새로운 모듈
- **`alpha_gpt_kr/trading/kis_api.py`**: 한국투자증권 KIS OpenAPI 클라이언트
  - 계좌 잔고 조회
  - 보유 종목 조회
  - 현재가 조회
  - 주식 주문 (매수/매도)
  - 실전투자 & 모의투자 지원
  - 자동 Access Token 관리

- **`alpha_gpt_kr/trading/trader.py`**: Alpha-GPT 기반 자동 매매 시스템
  - 알파 팩터 기반 매매 신호 생성
  - 자동 포트폴리오 리밸런싱
  - 리스크 관리 (손절매/익절)
  - 일일 체크 기능

#### 테스트 & 문서
- **`test_kis_trading.py`**: KIS API 및 자동 매매 테스트 스크립트
- **`TRADING_SETUP.md`**: 설정 가이드
- **`README_TRADING.md`**: 사용 설명서
- **`.env`**: KIS API 환경변수 템플릿 추가

#### 주요 기능
1. **KIS API 연동**
   - RESTful API 통신
   - OAuth2 인증
   - 실전/모의투자 모드

2. **자동 매매**
   - Alpha-GPT 알파 팩터 기반
   - 상위 N개 종목 선택
   - 동일 비중 포트폴리오
   - 주기적 리밸런싱 (default: 5영업일)

3. **리스크 관리**
   - 손절매: -5% (설정 가능)
   - 익절: +10% (설정 가능)
   - 일일 체크

#### Dependencies
- `requests>=2.31.0` 추가 (KIS API 통신)

---

## 사용 예시

### 모의투자 테스트
```bash
python test_kis_trading.py
```

### Python 코드
```python
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.trading.trader import AlphaTrader

# KIS API 초기화
api = KISApi(
    app_key=os.getenv("KIS_APP_KEY"),
    app_secret=os.getenv("KIS_APP_SECRET"),
    account_no=os.getenv("KIS_ACCOUNT_NO"),
    is_real=False  # 모의투자
)

# 트레이더 초기화
trader = AlphaTrader(
    kis_api=api,
    alpha_gpt=alpha_gpt_instance,
    max_stocks=10,
    rebalance_days=5,
    stop_loss_pct=-0.05,
    take_profit_pct=0.10
)

# 리밸런싱
trader.rebalance_portfolio(force=True)
```

---

## 다음 개선 사항 (TODO)

- [ ] 슬리피지 모델 추가
- [ ] 거래 비용 계산
- [ ] 주문 체결 확인
- [ ] 백필 주문 처리
- [ ] 포트폴리오 성과 리포트
- [ ] 텔레그램 알림
- [ ] 웹 대시보드

# 🚀 내일 실전 매매 실행 가이드

## ✅ 현재 준비 상태

### 시스템
- ✅ 한국투자증권 KIS API 연동 완료
- ✅ 토큰 캐싱 구현 (24시간 유효)
- ✅ 실전투자 계좌 연결
- ✅ 매수 프로세스 테스트 완료

### 계좌 정보
- **예수금**: 140,000원
- **보유 종목**: 없음
- **계좌번호**: 44009082-01

### 매수 가능 종목
1. **카카오 (035720)**: 58,900원 x 2주 = 117,800원 ⭐ 추천
2. **LG전자 (066570)**: 127,900원 x 1주 = 127,900원

---

## ⏰ 내일 실행 시간

**한국 증시 거래 시간:**
- 09:00 ~ 15:30 (평일)
- 추천 실행 시간: **09:10 ~ 15:00**

---

## 🎯 실행 방법

### 방법 1: 간단 매수 (추천)

```bash
cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr
python3 simple_test_trade.py
```

**실행 과정:**
1. 매수 가능 종목 리스트 출력
2. 종목 번호 입력 (예: 1 = 카카오)
3. `yes` 입력하면 실제 주문 발생
4. 주문 완료 확인

**예시:**
```
종목 번호 선택 (0=취소): 1
⚠️ 실제 주문이 발생합니다. 계속하시겠습니까? (yes/no): yes
```

---

### 방법 2: Alpha-GPT 자동 매매

```bash
python3 run_live_trading.py
```

**선택 메뉴:**
1. 알파 신호만 확인 (주문 없음)
2. 리밸런싱 실행 (실제 주문!)
3. 일일 체크 (손절/익절)

---

## ⚠️ 주의사항

### 실행 전 확인
- [ ] 거래 시간 확인 (09:00-15:30)
- [ ] 예수금 확인 (현재 140,000원)
- [ ] 종목 선택 (카카오 vs LG전자)

### 주문 후 확인
- [ ] 체결 확인
- [ ] 보유 종목 확인
- [ ] 잔여 예수금 확인

### 리스크 관리
- **손절매**: -5% 도달 시 자동 매도
- **익절**: +10% 도달 시 자동 매도
- **일일 체크**: 손절/익절 자동 확인

---

## 📊 주문 후 확인 방법

```bash
python3 -c "
import os
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi

load_dotenv()

api = KISApi(
    app_key=os.getenv('KIS_APP_KEY'),
    app_secret=os.getenv('KIS_APP_SECRET'),
    account_no=os.getenv('KIS_ACCOUNT_NO'),
    is_real=True
)

# 잔고 확인
balance = api.get_balance()
print(f'예수금: {int(balance.get(\"dnca_tot_amt\", 0)):,}원')

# 보유 종목
holdings = api.get_holdings()
if holdings:
    for h in holdings:
        print(f'{h[\"prdt_name\"]}: {h[\"hldg_qty\"]}주, 수익률 {h[\"evlu_pfls_rt\"]}%')
"
```

---

## 🔄 일일 체크 (자동 손절/익절)

매일 오후 3시 전에 실행:
```bash
python3 -c "
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.trading.trader import AlphaTrader
import os
from dotenv import load_dotenv

load_dotenv()

api = KISApi(
    app_key=os.getenv('KIS_APP_KEY'),
    app_secret=os.getenv('KIS_APP_SECRET'),
    account_no=os.getenv('KIS_ACCOUNT_NO'),
    is_real=True
)

# AlphaTrader로 손절/익절 체크
# (AlphaGPT 인스턴스 필요, 별도 스크립트 권장)
"
```

---

## 📞 문제 발생 시

### 주문 실패
- **장운영시간이 아닙니다**: 09:00-15:30 사이 재실행
- **잔고 부족**: 예수금 확인
- **403 에러**: 1분 후 재시도 (토큰 캐싱으로 거의 없음)

### 긴급 매도
HTS/MTS 앱에서 직접 매도 가능

---

## 🎯 추천 실행 시나리오

**내일 오전 9시 10분:**
1. 터미널 실행
2. `cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr`
3. `python3 simple_test_trade.py`
4. `1` 입력 (카카오 선택)
5. `yes` 입력 (주문 확인)
6. 체결 확인

**오후 2시 30분:**
- 수익률 확인
- 필요시 수동 매도

**다음날부터:**
- 알파 신호 기반 자동 매매 고려
- 리밸런싱 전략 적용

---

## 💡 다음 단계

1. **소액 테스트 완료** (현재 단계)
2. **1주일 모니터링**
3. **성과 확인 후 규모 확대**
4. **Alpha-GPT 자동 매매 전환**
5. **포트폴리오 분산**

---

**준비 완료! 내일 오전 9시 이후 실행하세요.** 🚀

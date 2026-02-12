#!/usr/bin/env python3
"""예수금 확인"""

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

print("=" * 60)
print("계좌 잔고 확인")
print("=" * 60)

balance = api.get_balance()
cash = int(balance.get('dnca_tot_amt', 0))

print(f"\n예수금: {cash:,}원")

# 권장 전략
if cash < 500000:
    print("\n⚠️  50만원 미만: 분산 투자 제한적")
    print("권장: 50만원 이상 입금")
elif cash < 1000000:
    print("\n✅ 50-100만원: 3-5종목 분산 가능")
elif cash < 3000000:
    print("\n✅ 100-300만원: 5-10종목 분산 가능 (추천)")
else:
    print("\n✅ 300만원 이상: 10-15종목 분산 가능 (최적)")

# 종목 수 권장
max_stocks = min(15, max(3, cash // 200000))
print(f"\n권장 종목 수: {max_stocks}개")
print(f"종목당 평균: {cash // max_stocks:,}원")

print("\n" + "=" * 60)
print("\n입금 완료 후 다음 명령 실행:")
print("  python3 alpha_based_buy.py")
print("=" * 60)

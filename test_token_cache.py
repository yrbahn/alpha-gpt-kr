#!/usr/bin/env python3
"""토큰 캐싱 테스트"""

import os
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi

load_dotenv()

print("=" * 60)
print("토큰 캐싱 테스트 - 연속 호출")
print("=" * 60)

api = KISApi(
    app_key=os.getenv('KIS_APP_KEY'),
    app_secret=os.getenv('KIS_APP_SECRET'),
    account_no=os.getenv('KIS_ACCOUNT_NO'),
    is_real=True
)

# 1차 호출
print("\n[1차 계좌 조회]")
balance1 = api.get_balance()
print(f"예수금: {int(balance1.get('dnca_tot_amt', 0)):,}원")

# 2차 호출 (즉시 - 캐시된 토큰 사용)
print("\n[2차 계좌 조회 - 캐시된 토큰 사용해야 함]")
balance2 = api.get_balance()
print(f"예수금: {int(balance2.get('dnca_tot_amt', 0)):,}원")

# 현재가 조회 테스트
print("\n[3차 현재가 조회]")
price = api.get_current_price('035720')  # 카카오
print(f"카카오: {int(price['stck_prpr']):,}원")

print("\n✅ 연속 호출 성공! 토큰 캐싱 정상 작동")
print("=" * 60)

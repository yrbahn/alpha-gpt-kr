#!/usr/bin/env python3
"""
메디톡스 전량 매수 스크립트
- 기존 보유 종목 전량 매도
- 메디톡스 (086900) 전량 매수
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from alpha_gpt_kr.trading.kis_api import KISApi

TARGET_TICKER = '086900'
TARGET_NAME = '메디톡스'


def main():
    print("=" * 60)
    print(f"🚀 메디톡스 전량 매수 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # KIS API 초기화
    kis = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO')
    )
    
    # 1. 현재 보유 종목 확인
    print("\n📊 1. 현재 보유 종목 확인...")
    holdings = kis.get_holdings()
    
    if holdings:
        print(f"  보유 종목 {len(holdings)}개:")
        for h in holdings:
            print(f"    - {h['ticker']} {h['name']}: {h['qty']}주")
    else:
        print("  보유 종목 없음")
    
    # 2. 기존 보유 종목 전량 매도
    if holdings:
        print("\n📤 2. 기존 보유 종목 전량 매도...")
        
        if '--execute' in sys.argv:
            for h in holdings:
                if h['qty'] > 0:
                    print(f"  매도: {h['ticker']} {h['name']} {h['qty']}주...")
                    try:
                        result = kis.sell(h['ticker'], h['qty'])
                        print(f"    ✅ 매도 주문 완료")
                    except Exception as e:
                        print(f"    ❌ 매도 실패: {e}")
        else:
            print("  (--execute 옵션 필요)")
    
    # 3. 잔고 확인
    print("\n💰 3. 잔고 확인...")
    balance = kis.get_balance()
    print(f"  예수금: {balance:,.0f}원")
    
    # 4. 메디톡스 현재가 조회
    print(f"\n📈 4. {TARGET_NAME} 현재가 조회...")
    price = kis.get_current_price(TARGET_TICKER)
    print(f"  현재가: {price:,.0f}원")
    
    # 5. 매수 수량 계산
    qty = int(balance / price)
    total_amount = qty * price
    
    print(f"\n🧮 5. 매수 계획")
    print(f"  매수 수량: {qty}주")
    print(f"  예상 금액: {total_amount:,.0f}원")
    print(f"  잔여 예수금: {balance - total_amount:,.0f}원")
    
    # 6. 메디톡스 매수
    print(f"\n📥 6. {TARGET_NAME} 매수...")
    
    if '--execute' in sys.argv:
        if qty > 0:
            try:
                result = kis.buy(TARGET_TICKER, qty)
                print(f"  ✅ 매수 주문 완료: {qty}주")
            except Exception as e:
                print(f"  ❌ 매수 실패: {e}")
        else:
            print(f"  ⚠️ 매수 가능 수량 없음 (잔고 부족)")
    else:
        print("  (--execute 옵션 필요)")
    
    print("\n" + "=" * 60)
    if '--execute' in sys.argv:
        print("✅ 완료!")
    else:
        print("⚠️ 테스트 모드 (실제 매매: --execute 옵션 추가)")
    print("=" * 60)


if __name__ == "__main__":
    main()

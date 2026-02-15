#!/usr/bin/env python3
"""
보유 종목 전량 매도
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import argparse

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.trading.kis_api import KISApi

# 환경 변수 로드
load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='시뮬레이션만 (실제 주문 X)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Alpha-GPT-KR: Sell All Holdings")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE TRADING'}")
    print("=" * 60)
    
    # KIS API 초기화
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 보유 종목 조회
    holdings = api.get_holdings()
    
    if not holdings:
        print("\n❌ No holdings found!")
        return
    
    print(f"\n📊 Current holdings: {len(holdings)}개")
    print("\n보유 종목:")
    
    total_value = 0
    sell_orders = []
    
    for h in holdings:
        ticker = h['pdno']  # 종목코드
        name = h['prdt_name']  # 종목명
        qty = int(h['hldg_qty'])  # 보유수량
        avg_price = float(h['pchs_avg_pric'])  # 평균매입가
        current_price = float(h['prpr'])  # 현재가
        eval_amt = float(h['evlu_amt'])  # 평가금액
        profit_loss = float(h['evlu_pfls_amt'])  # 평가손익
        profit_rate = float(h['evlu_pfls_rt'])  # 수익률
        
        total_value += eval_amt
        
        print(f"   {ticker} ({name}): {qty}주")
        print(f"      매입가: {int(avg_price):,}원 | 현재가: {int(current_price):,}원")
        print(f"      평가액: {int(eval_amt):,}원 | 손익: {int(profit_loss):,}원 ({profit_rate:+.2f}%)")
        
        sell_orders.append({
            'ticker': ticker,
            'name': name,
            'qty': qty,
            'current_price': current_price,
            'eval_amt': eval_amt
        })
    
    print(f"\n💰 총 평가금액: {int(total_value):,}원")
    
    if args.dry_run:
        print("\n✅ DRY RUN mode - no actual orders")
        print("\n📋 Would sell:")
        for order in sell_orders:
            print(f"   {order['ticker']} ({order['name']}): {order['qty']}주")
        return
    
    # 실전 매도
    print("\n⚠️  WARNING: 모든 보유 종목을 시장가로 매도합니다!")
    confirm = input("계속하시겠습니까? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("❌ 취소되었습니다.")
        return
    
    print("\n🚀 Starting sell orders...")
    
    results = []
    
    for order in sell_orders:
        ticker = order['ticker']
        name = order['name']
        qty = order['qty']
        
        print(f"\n📉 Selling {ticker} ({name}): {qty}주")
        
        try:
            sell_order = api.sell_stock(ticker, qty, order_type="01")
            results.append({
                'ticker': ticker,
                'name': name,
                'qty': qty,
                'status': 'success',
                'order_no': sell_order.get('ODNO', '')
            })
            print(f"   ✅ Order placed: {sell_order.get('ODNO', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'ticker': ticker,
                'name': name,
                'qty': qty,
                'status': 'failed',
                'error': str(e)
            })
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 Sell Results")
    print("=" * 60)
    
    success_count = len([r for r in results if r['status'] == 'success'])
    print(f"\n✅ Success: {success_count}/{len(results)}")
    
    if success_count > 0:
        print("\n💡 매도 완료 후 잔고를 확인하세요:")
        print("   python3 check_balance.py")
    
    print("\n🎉 Sell all completed!")

if __name__ == "__main__":
    main()

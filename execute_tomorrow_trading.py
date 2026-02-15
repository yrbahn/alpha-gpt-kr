#!/usr/bin/env python3
"""
내일 아침 09:00 매매 실행
선정된 8개 종목 매수
"""

import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.trading.kis_api import KISApi

load_dotenv()

def main():
    print("=" * 80)
    print("Alpha-GPT: 15일 Forward 매매 실행")
    print("=" * 80)
    print()
    
    # CSV 파일 읽기
    csv_file = 'selected_stocks_tomorrow.csv'
    
    if not Path(csv_file).exists():
        print(f"❌ {csv_file} 파일이 없습니다!")
        print("   먼저 calculate_best_alpha.py를 실행하세요.")
        return
    
    df = pd.read_csv(csv_file)
    
    print("📋 매수 대상 종목:")
    print(df.to_string(index=False))
    
    # 상위 8개만
    top_8 = df.head(8)
    
    print(f"\n💰 총 투자 금액: 5,000,000원")
    print(f"   종목 수: {len(top_8)}개")
    print(f"   종목당: {5_000_000 // len(top_8):,}원")
    
    # 확인
    confirm = input("\n계속하시겠습니까? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("❌ 취소되었습니다.")
        return
    
    # KIS API 초기화
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 잔고 조회
    balance = api.get_balance()
    available_cash = balance['dnca_tot_amt']
    
    print(f"\n💵 예수금: {available_cash:,}원")
    
    if available_cash < 5_000_000:
        print(f"⚠️  Warning: 예수금 부족 (필요: 5,000,000원)")
        proceed = input("그래도 진행하시겠습니까? (yes/no): ")
        if proceed.lower() != 'yes':
            print("❌ 취소되었습니다.")
            return
    
    # 매수 실행
    print("\n🚀 매수 시작...")
    
    amount_per_stock = 5_000_000 // len(top_8)
    results = []
    
    for idx, row in top_8.iterrows():
        ticker = row['종목코드']
        name = row['종목명']
        
        # 현재가 조회
        try:
            price_info = api.get_current_price(ticker)
            current_price = int(price_info['stck_prpr'])  # 현재가
            
            # 거래정지 확인
            status_code = price_info.get('iscd_stat_cls_code', '00')
            if status_code == '58':
                print(f"\n⚠️  {ticker} ({name}): 거래정지 - 건너뜀")
                results.append({
                    'ticker': ticker,
                    'name': name,
                    'status': 'skipped',
                    'reason': '거래정지'
                })
                continue
            
            # 수량 계산
            qty = int(amount_per_stock / current_price)
            
            if qty == 0:
                print(f"\n⏭️  {ticker} ({name}): 가격 너무 높음 ({current_price:,}원) - 건너뜀")
                results.append({
                    'ticker': ticker,
                    'name': name,
                    'status': 'skipped',
                    'reason': '가격 높음'
                })
                continue
            
            print(f"\n📈 매수: {ticker} ({name})")
            print(f"   수량: {qty}주 × {current_price:,}원 = {qty * current_price:,}원")
            
            # 시장가 매수
            order = api.buy_market(ticker, qty)
            
            results.append({
                'ticker': ticker,
                'name': name,
                'qty': qty,
                'price': current_price,
                'status': 'success',
                'order_no': order.get('ODNO', '')
            })
            
            print(f"   ✅ 주문 완료: {order.get('ODNO', 'N/A')}")
            
        except Exception as e:
            print(f"\n❌ 에러: {ticker} ({name}) - {e}")
            results.append({
                'ticker': ticker,
                'name': name,
                'status': 'failed',
                'error': str(e)
            })
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 매수 결과")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    success_count = len([r for r in results if r.get('status') == 'success'])
    print(f"\n✅ 성공: {success_count}/{len(results)}")
    
    print("\n🎉 매매 완료!")
    print("\n⏰ 15일 후 (2026-02-28경) 리밸런싱 예정")

if __name__ == "__main__":
    main()

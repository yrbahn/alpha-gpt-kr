#!/usr/bin/env python3
"""
소액 실전 테스트 - 간단한 모멘텀 전략
"""

import os
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi
from loguru import logger

load_dotenv()

# 테스트할 저가 종목 리스트 (수동 선택)
# 한국 증시 저가 우량주/중소형주
TEST_TICKERS = [
    '005930',  # 삼성전자
    '000660',  # SK하이닉스
    '035420',  # NAVER
    '005380',  # 현대차
    '051910',  # LG화학
    '006400',  # 삼성SDI
    '035720',  # 카카오
    '028260',  # 삼성물산
    '012330',  # 현대모비스
    '066570',  # LG전자
]

def main():
    print("=" * 60)
    print("소액 실전 테스트 - 5만원으로 가능한 종목 찾기")
    print("=" * 60)
    
    # KIS API 초기화
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 계좌 확인
    print("\n[계좌 정보]")
    balance = api.get_balance()
    cash = int(balance.get('dnca_tot_amt', 0))
    print(f"예수금: {cash:,}원")
    
    if cash < 10000:
        print("\n⚠️ 예수금 부족: 최소 1만원 이상 필요")
        return
    
    # 저가 종목 찾기
    print(f"\n[5만원 이하 종목 검색]")
    print("-" * 60)
    
    affordable = []
    
    for ticker in TEST_TICKERS:
        try:
            price_info = api.get_current_price(ticker)
            price = int(price_info['stck_prpr'])
            name = price_info.get('prdy_vrss_sign', ticker)
            
            if price <= cash:
                qty = cash // price
                cost = price * qty
                affordable.append({
                    'ticker': ticker,
                    'price': price,
                    'qty': qty,
                    'cost': cost
                })
                print(f"✓ {ticker}: {price:>8,}원 x {qty}주 = {cost:>8,}원")
            else:
                print(f"✗ {ticker}: {price:>8,}원 (예수금 초과)")
                
        except Exception as e:
            logger.warning(f"✗ {ticker}: {e}")
    
    if not affordable:
        print("\n⚠️ 예수금으로 살 수 있는 종목이 없습니다.")
        print("\n대안:")
        print("1. 예수금 추가 입금")
        print("2. 더 저가 종목 찾기")
        print("3. 모의투자 계좌 사용")
        return
    
    # 추천
    print("\n" + "=" * 60)
    print("✅ 매수 가능한 종목")
    print("=" * 60)
    
    for i, stock in enumerate(affordable, 1):
        print(f"\n{i}. {stock['ticker']}")
        print(f"   현재가: {stock['price']:,}원")
        print(f"   매수 가능: {stock['qty']}주 = {stock['cost']:,}원")
    
    # 매수 진행
    print("\n" + "=" * 60)
    print("매수하시겠습니까?")
    print("=" * 60)
    
    choice = input("\n종목 번호 선택 (0=취소): ").strip()
    
    if choice == '0' or not choice.isdigit():
        print("취소되었습니다.")
        return
    
    idx = int(choice) - 1
    if idx < 0 or idx >= len(affordable):
        print("잘못된 선택입니다.")
        return
    
    selected = affordable[idx]
    
    print(f"\n선택: {selected['ticker']}")
    print(f"매수: {selected['qty']}주 @ {selected['price']:,}원")
    print(f"총액: {selected['cost']:,}원")
    
    confirm = input("\n⚠️ 실제 주문이 발생합니다. 계속하시겠습니까? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("취소되었습니다.")
        return
    
    # 실제 매수
    try:
        print("\n[주문 실행 중...]")
        result = api.buy_stock(
            ticker=selected['ticker'],
            qty=selected['qty'],
            order_type="01"  # 시장가
        )
        
        print("\n✅ 주문 완료!")
        print(f"주문번호: {result.get('ODNO', 'N/A')}")
        print(f"주문시각: {result.get('ORD_TMD', 'N/A')}")
        
        print("\n📊 주문 후 계좌 확인:")
        balance2 = api.get_balance()
        print(f"예수금: {int(balance2.get('dnca_tot_amt', 0)):,}원")
        
        holdings = api.get_holdings()
        if holdings:
            for h in holdings:
                if h['pdno'] == selected['ticker']:
                    print(f"\n보유: {h['prdt_name']} {h['hldg_qty']}주")
                    print(f"평가금액: {int(h['evlu_amt']):,}원")
        
    except Exception as e:
        print(f"\n❌ 주문 실패: {e}")
        logger.error(f"Order failed: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

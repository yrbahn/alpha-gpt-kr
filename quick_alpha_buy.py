#!/usr/bin/env python3
"""
빠른 알파 기반 매수
- GP 진화 최고 알파 개념 사용 (26일 모멘텀)
- DB 없이 KIS API만 사용
"""

import os
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi
from loguru import logger

load_dotenv()

# 시가총액 상위 대표 종목 리스트 (수동 선택)
# GP 알파: ops.ts_delta(close, 26) - 26일 모멘텀
# 여기서는 한국 증시 대표 종목들로 테스트
CANDIDATE_TICKERS = [
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
    '105560',  # KB금융
    '055550',  # 신한지주
    '000270',  # 기아
    '096770',  # SK이노베이션
    '207940',  # 삼성바이오로직스
    '068270',  # 셀트리온
    '005490',  # POSCO홀딩스
    '373220',  # LG에너지솔루션
    '086790',  # 하나금융지주
    '323410',  # 카카오뱅크
]

def main():
    print("=" * 60)
    print("🚀 빠른 알파 기반 포트폴리오 구성")
    print("=" * 60)
    print("전략: GP 진화 최고 알파 개념 (모멘텀)")
    print()
    
    # KIS API
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 계좌 확인
    balance = api.get_balance()
    cash = int(balance.get('dnca_tot_amt', 0))
    print(f"예수금: {cash:,}원")
    
    target_stocks = min(15, max(5, cash // 300000))
    print(f"목표 종목 수: {target_stocks}개")
    print(f"종목당 평균: {cash // target_stocks:,}원\n")
    
    # 종목 분석
    print("[종목 분석 중...]")
    print("-" * 60)
    
    stocks = []
    
    for ticker in CANDIDATE_TICKERS:
        try:
            price_info = api.get_current_price(ticker)
            price = int(price_info['stck_prpr'])
            
            # 간단한 모멘텀 지표 (전일대비)
            change_rate = float(price_info.get('prdy_ctrt', 0))
            
            # 매수 가능 여부
            per_stock_budget = cash // target_stocks
            if price <= per_stock_budget:
                qty = per_stock_budget // price
                cost = price * qty
                
                stocks.append({
                    'ticker': ticker,
                    'price': price,
                    'qty': qty,
                    'cost': cost,
                    'change_rate': change_rate
                })
                
                print(f"✓ {ticker}: {price:>9,}원, 변동: {change_rate:>6}%")
            else:
                print(f"✗ {ticker}: {price:>9,}원 (예산 초과)")
                
        except Exception as e:
            logger.warning(f"✗ {ticker}: {e}")
            continue
    
    if len(stocks) < target_stocks:
        print(f"\n⚠️  목표 {target_stocks}개 중 {len(stocks)}개만 가능")
        print("그래도 진행할까요?")
    
    # 변동률 순으로 정렬 (모멘텀)
    stocks.sort(key=lambda x: x['change_rate'], reverse=True)
    
    # 상위 target_stocks개 선택
    selected = stocks[:target_stocks]
    
    print("\n" + "=" * 60)
    print(f"✅ 선택된 포트폴리오 ({len(selected)}개 종목)")
    print("=" * 60)
    
    total_cost = sum(s['cost'] for s in selected)
    
    for i, stock in enumerate(selected, 1):
        print(f"\n{i:2d}. {stock['ticker']}")
        print(f"    가격: {stock['price']:,}원")
        print(f"    수량: {stock['qty']}주")
        print(f"    금액: {stock['cost']:,}원")
        print(f"    변동: {stock['change_rate']}%")
    
    print(f"\n총 투자금액: {total_cost:,}원")
    print(f"잔여 예수금: {cash - total_cost:,}원")
    
    # 매수 확인
    print("\n" + "=" * 60)
    confirm = input(f"\n⚠️  {len(selected)}개 종목을 매수하시겠습니까? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("취소되었습니다.")
        return
    
    # 일괄 매수
    print("\n[일괄 매수 시작...]")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for stock in selected:
        try:
            print(f"\n매수 중: {stock['ticker']} {stock['qty']}주...")
            result = api.buy_stock(
                ticker=stock['ticker'],
                qty=stock['qty'],
                order_type="01"  # 시장가
            )
            print(f"  ✅ 주문 완료 (주문번호: {result.get('ODNO', 'N/A')})")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 실패: {e}")
            fail_count += 1
    
    # 결과
    print("\n" + "=" * 60)
    print("📊 매수 결과")
    print("=" * 60)
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    
    # 계좌 확인
    print("\n[최종 계좌 확인]")
    balance2 = api.get_balance()
    print(f"예수금: {int(balance2.get('dnca_tot_amt', 0)):,}원")
    
    holdings = api.get_holdings()
    if holdings:
        print(f"보유 종목: {len(holdings)}개")
        for h in holdings:
            print(f"  {h['prdt_name']}: {h['hldg_qty']}주")
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

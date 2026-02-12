#!/usr/bin/env python3
"""
현재 포트폴리오 vs Alpha 기준 비교
- GP 진화 결과 활용
- 실제 작동하는 코드
"""

import os
import pandas as pd
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from loguru import logger

load_dotenv()

def main():
    print("=" * 60)
    print("📊 현재 포트폴리오 vs GP 알파 기준 비교")
    print("=" * 60)
    
    # KIS API
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 현재 보유 종목
    holdings = api.get_holdings()
    current_tickers = [h['pdno'] for h in holdings]
    
    print(f"\n현재 보유: {len(holdings)}개 종목")
    for h in holdings:
        print(f"  {h['prdt_name']} ({h['pdno']}): {h['hldg_qty']}주")
    
    # 데이터 로딩
    print("\n[데이터 로딩...]")
    try:
        loader = PostgresDataLoader(
            host="192.168.0.248",
            port=5432,
            database="marketsense",
            user="yrbahn",
            password="1234"
        )
        
        data = loader.load_data(
            start_date="2024-01-01",
            end_date="2025-02-11",
            universe=None
        )
        
        print(f"✅ 데이터 로드: {len(data['close'].columns)}개 종목")
        
    except Exception as e:
        print(f"❌ DB 로딩 실패: {e}")
        return
    
    # 알파 계산 (GP 최고 알파)
    print("\n[GP 최고 알파 계산...]")
    print("ops.ts_delta(close, 26)")
    
    close = data['close']
    volume = data['volume']
    
    # 26일 델타 계산
    alpha_values = ops.ts_delta(close, 26)
    
    # 최신 날짜
    latest_date = alpha_values.index[-1]
    latest_alpha = alpha_values.loc[latest_date]
    
    print(f"계산 완료: {latest_date}")
    
    # 상위 종목
    top_tickers = latest_alpha.sort_values(ascending=False).head(20)
    
    print(f"\n📈 GP 알파 상위 20개:")
    for i, (ticker, score) in enumerate(top_tickers.items(), 1):
        in_portfolio = "✓" if ticker in current_tickers else " "
        print(f"  {i:2d}. [{in_portfolio}] {ticker}: {score:+.4f}")
    
    # 현재 포트폴리오 알파 점수
    print(f"\n📊 현재 포트폴리오 알파 분석:")
    
    portfolio_scores = []
    for ticker in current_tickers:
        if ticker in latest_alpha.index:
            score = latest_alpha[ticker]
            rank = (latest_alpha > score).sum() + 1
            portfolio_scores.append({
                'ticker': ticker,
                'score': score,
                'rank': rank
            })
    
    df_portfolio = pd.DataFrame(portfolio_scores).sort_values('rank')
    
    print(f"\n{'종목':<10} {'알파 점수':<12} {'순위':<8}")
    print("-" * 35)
    for _, row in df_portfolio.iterrows():
        print(f"{row['ticker']:<10} {row['score']:>+10.4f}  {int(row['rank']):>5d}위")
    
    avg_rank = df_portfolio['rank'].mean()
    median_rank = df_portfolio['rank'].median()
    
    print(f"\n평균 순위: {avg_rank:.0f}위")
    print(f"중앙값 순위: {median_rank:.0f}위")
    
    # 비교 분석
    print("\n" + "=" * 60)
    print("📊 분석 결과")
    print("=" * 60)
    
    top20_set = set(top_tickers.index)
    current_set = set(current_tickers)
    
    overlap = top20_set & current_set
    missing = top20_set - current_set
    extra = current_set - top20_set
    
    print(f"\n✅ 알파 상위 20개 중 보유: {len(overlap)}개 ({len(overlap)/20*100:.0f}%)")
    for ticker in sorted(overlap):
        rank = (latest_alpha > latest_alpha[ticker]).sum() + 1
        print(f"   {ticker}: {rank}위")
    
    print(f"\n⚠️  알파 상위인데 미보유: {len(missing)}개")
    for ticker in list(missing)[:5]:
        rank = (latest_alpha > latest_alpha[ticker]).sum() + 1
        score = latest_alpha[ticker]
        print(f"   {ticker}: {rank}위 (점수: {score:+.4f})")
    
    print(f"\n🔴 보유했지만 상위 아님: {len(extra)}개")
    for ticker in sorted(extra):
        rank = (latest_alpha > latest_alpha[ticker]).sum() + 1
        score = latest_alpha[ticker]
        print(f"   {ticker}: {rank}위 (점수: {score:+.4f})")
    
    # 결론
    print("\n" + "=" * 60)
    print("💡 결론")
    print("=" * 60)
    
    if avg_rank < 100:
        print("✅ 현재 포트폴리오 알파 품질: 우수")
    elif avg_rank < 250:
        print("⚠️  현재 포트폴리오 알파 품질: 보통")
    else:
        print("🔴 현재 포트폴리오 알파 품질: 개선 필요")
    
    print(f"\n현재 포트폴리오는 대형주 위주로 구성되어 있습니다.")
    print(f"GP 알파 기준으로는 평균 {avg_rank:.0f}위 수준입니다.")
    
    if len(overlap) < 10:
        print(f"\n📌 제안: 알파 상위 종목으로 일부 교체 고려")
        print("교체 후보:")
        for ticker in list(missing)[:3]:
            rank = (latest_alpha > latest_alpha[ticker]).sum() + 1
            print(f"  + {ticker} ({rank}위)")
        print("매도 후보:")
        worst = df_portfolio.tail(3)
        for _, row in worst.iterrows():
            print(f"  - {row['ticker']} ({int(row['rank'])}위)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

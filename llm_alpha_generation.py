#!/usr/bin/env python3
"""
🧠 LLM 기반 복잡한 알파 생성
GPT-4가 제안하는 정교한 팩터 조합
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.backtest.engine import BacktestEngine
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from loguru import logger

load_dotenv()

# OpenAI 클라이언트
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def ask_gpt_for_alphas():
    """GPT-4에게 복잡한 알파 요청"""
    
    prompt = """You are a quantitative trading expert specializing in Korean stock market (KRX).

Current situation:
- Simple 26-day momentum (ts_delta(close, 26)) only achieves IC: 0.0045, Sharpe: 0.57
- This is too weak for profitable trading
- We need MORE COMPLEX and EFFECTIVE alpha factors

Available operators:
- ts_delta(data, d): difference from d days ago
- ts_mean(data, d): d-day moving average
- ts_std(data, d): d-day standard deviation
- ts_max/ts_min(data, d): d-day max/min
- ts_rank(data, d): rank within d-day window
- ts_corr(x, y, d): d-day correlation
- rank(data): cross-sectional rank
- scale(data): normalize to sum=1

Available data:
- close: closing price
- volume: trading volume
- open, high, low, vwap

Task: Generate 10 COMPLEX alpha expressions that combine multiple factors:
1. Momentum + Volume (e.g., strong momentum with high volume)
2. Volatility + Trend (e.g., low volatility uptrend)
3. Mean reversion + Volume
4. Multi-timeframe momentum
5. Volume-weighted price patterns
6. Correlation-based signals
7. Rank-based combinations
8. Volatility-adjusted momentum
9. Price-volume divergence
10. Complex mathematical combinations

Requirements:
- Each alpha should use 2-4 operators
- Combine different timeframes (5, 10, 20, 26 days)
- Use both price and volume
- Be creative but logical

Format: Return ONLY valid Python expressions using ops.xxx() syntax.
Example: "ops.rank(ops.ts_delta(close, 20) * ops.ts_mean(volume, 10))"

Generate 10 complex alpha expressions:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )
    
    content = response.choices[0].message.content
    
    # 표현식 추출
    alphas = []
    for line in content.split('\n'):
        line = line.strip()
        if 'ops.' in line and '(' in line:
            # 숫자나 설명 제거
            if '. ' in line:
                line = line.split('. ', 1)[1]
            if '#' in line:
                line = line.split('#')[0]
            line = line.strip().strip('"').strip("'")
            if line and line.startswith('ops.'):
                alphas.append(line)
    
    return alphas


def backtest_alpha(expression, close, volume, returns, tickers):
    """알파 백테스트"""
    try:
        # 표현식 평가
        logger.info(f"테스트 중: {expression}")
        alpha_values = eval(expression)
        
        # 백테스트
        engine = BacktestEngine(
            universe=tickers,
            price_data=close,
            return_data=returns
        )
        
        result = engine.backtest(
            alpha_values=alpha_values,
            long_short=True,
            top_pct=0.2,
            transaction_cost=0.003
        )
        
        return {
            'expression': expression,
            'ic': result.ic,
            'sharpe': result.sharpe,
            'annual_return': result.total_return * (252 / len(returns)) * 100,
            'result': result
        }
        
    except Exception as e:
        logger.warning(f"실패: {expression} - {e}")
        return None


def main():
    print("=" * 60)
    print("🧠 LLM 기반 복잡한 알파 생성")
    print("=" * 60)
    
    # 1. GPT-4에게 알파 요청
    print("\n[1] GPT-4에게 복잡한 알파 요청 중...")
    llm_alphas = ask_gpt_for_alphas()
    
    print(f"✅ GPT-4가 {len(llm_alphas)}개 알파 생성:")
    for i, alpha in enumerate(llm_alphas, 1):
        print(f"  {i}. {alpha}")
    
    # 2. 데이터 로딩
    print("\n[2] 데이터 로딩...")
    loader = PostgresDataLoader(
        host="192.168.0.248",
        port=5432,
        database="marketsense",
        user="yrbahn",
        password="1234"
    )
    
    data = loader.load_data(
        start_date="2023-01-01",
        end_date="2025-02-11",
        universe=None
    )
    
    close = data['close']
    volume = data['volume']
    returns = close.pct_change()
    tickers = close.columns.tolist()
    
    print(f"✅ {len(tickers)}개 종목, {len(close)}일")
    
    # 3. 백테스트
    print("\n[3] LLM 알파 백테스트 중...")
    print("-" * 60)
    
    results = []
    
    for alpha_expr in llm_alphas:
        result = backtest_alpha(alpha_expr, close, volume, returns, tickers)
        if result:
            results.append(result)
            print(f"\n✓ {result['expression'][:80]}...")
            print(f"  IC: {result['ic']:.4f}, Sharpe: {result['sharpe']:.2f}, 연수익: {result['annual_return']:.1f}%")
    
    # 4. 결과 정리
    print("\n" + "=" * 60)
    print("📊 결과 요약")
    print("=" * 60)
    
    if not results:
        print("⚠️ 성공한 알파 없음")
        return
    
    # 정렬
    results.sort(key=lambda x: x['ic'], reverse=True)
    
    print(f"\n🏆 Top 5 알파 (IC 순):\n")
    
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. IC: {r['ic']:+.4f} | Sharpe: {r['sharpe']:+.2f} | 연수익: {r['annual_return']:+.1f}%")
        print(f"   {r['expression']}")
        print()
    
    # 5. 기준 비교
    print("=" * 60)
    print("📊 기준 비교")
    print("=" * 60)
    
    best = results[0]
    baseline_ic = 0.0045
    baseline_sharpe = 0.57
    
    print(f"\n기존 (26일 모멘텀):")
    print(f"  IC: {baseline_ic:.4f}")
    print(f"  Sharpe: {baseline_sharpe:.2f}")
    
    print(f"\n🧠 LLM 최고 알파:")
    print(f"  IC: {best['ic']:.4f} ({(best['ic'] - baseline_ic) / baseline_ic * 100:+.0f}% 개선)")
    print(f"  Sharpe: {best['sharpe']:.2f} ({(best['sharpe'] - baseline_sharpe) / baseline_sharpe * 100:+.0f}% 개선)")
    print(f"  표현식: {best['expression']}")
    
    # 6. 저장
    print("\n" + "=" * 60)
    print("💾 결과 저장 중...")
    
    with open('llm_alpha_results.txt', 'w') as f:
        f.write("🧠 LLM 생성 알파 백테스트 결과\n")
        f.write("=" * 60 + "\n\n")
        
        for i, r in enumerate(results, 1):
            f.write(f"{i}. IC: {r['ic']:+.4f} | Sharpe: {r['sharpe']:+.2f}\n")
            f.write(f"   {r['expression']}\n\n")
    
    print("✅ llm_alpha_results.txt 저장 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()

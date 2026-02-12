#!/usr/bin/env python3
"""
Alpha-GPT 논문 방식 (간단 버전)
LLM이 직접 알파 표현식을 생성하고 평가
"""

import sys
import os
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import openai

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators

# 환경 변수 로드
load_dotenv()

# DB 연결
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

# 상위 500개 종목 + 가격 데이터 로드
def load_market_data():
    """시가총액 상위 500개 종목 데이터 로드 (간단 버전)"""
    print("📊 데이터 로드 중...")
    
    conn = get_db_connection()
    
    # 상위 500개 종목
    query_stocks = """
        SELECT DISTINCT ON (s.ticker)
            s.id, s.ticker, s.name
        FROM stocks s
        JOIN price_data p ON s.id = p.stock_id
        WHERE s.is_active = true
        AND p.date = (SELECT MAX(date) FROM price_data)
        AND p.close IS NOT NULL AND p.volume IS NOT NULL
        ORDER BY s.ticker, (p.close * p.volume) DESC
        LIMIT 100
    """
    
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    
    # 가격 데이터 (최근 6개월)
    stock_id_list = ', '.join(map(str, stock_ids))
    query_prices = f"""
        SELECT 
            s.ticker,
            p.date,
            p.close,
            p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    conn.close()
    
    # 피벗 테이블로 변환 (종목 x 날짜)
    close_pivot = price_df.pivot(index='date', columns='ticker', values='close')
    volume_pivot = price_df.pivot(index='date', columns='ticker', values='volume')
    
    print(f"✅ {len(close_pivot.columns)}개 종목, {len(close_pivot)}일 데이터 로드")
    
    return {
        'close': close_pivot,
        'volume': volume_pivot,
        'returns': close_pivot.pct_change()
    }

# LLM으로 알파 생성
def generate_alphas_with_llm(idea, num_alphas=5):
    """LLM에게 알파 표현식 생성 요청 (논문 방식)"""
    
    print(f"\n🤖 LLM이 알파 생성 중... (GPT-4)")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""당신은 퀀트 개발자입니다. 다음 트레이딩 아이디어를 바탕으로 알파 표현식을 생성하세요.

트레이딩 아이디어:
{idea}

사용 가능한 데이터:
- close: 종가 (pandas DataFrame)
- volume: 거래량 (pandas DataFrame)
- returns: 수익률 (pandas DataFrame)

사용 가능한 연산자 (AlphaOperators 클래스):
- ts_delta(x, period): 현재값 - N일 전 값
- ts_mean(x, window): 이동 평균
- ts_std(x, window): 이동 표준편차
- ts_rank(x, window): 순위 (0~1)
- ts_corr(x, y, window): 상관계수
- ts_max(x, window): 이동 최대값
- ts_min(x, window): 이동 최소값

규칙:
1. 표현식은 pandas DataFrame을 입력받고 DataFrame을 반환해야 함
2. AlphaOperators의 메서드는 정적 메서드로 호출 (AlphaOperators.ts_mean(close, 20))
3. 실제로 실행 가능한 Python 코드여야 함
4. 복잡하고 창의적인 조합을 사용하세요

{num_alphas}개의 서로 다른 알파 표현식을 생성하세요.
각 표현식은 한 줄로, 다음 형식으로 출력하세요:

ALPHA_1: [표현식]
ALPHA_2: [표현식]
...

예시:
ALPHA_1: AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 20) / AlphaOperators.ts_std(close, 20), 10)
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a quantitative researcher creating alpha expressions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=2000
    )
    
    # 응답 파싱
    content = response.choices[0].message.content
    
    alphas = []
    for line in content.split('\n'):
        if line.strip().startswith('ALPHA_'):
            # "ALPHA_1: 표현식" 형태에서 표현식 추출
            parts = line.split(':', 1)
            if len(parts) == 2:
                expr = parts[1].strip()
                alphas.append(expr)
    
    print(f"✅ {len(alphas)}개 알파 생성 완료")
    
    return alphas

# 알파 평가 (IC 계산)
def evaluate_alpha(alpha_expr, data):
    """알파 표현식을 평가하여 IC (Information Coefficient) 계산"""
    
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns'].shift(-1)  # 다음 날 수익률
        
        # 알파 계산
        alpha_values = eval(alpha_expr)
        
        # IC 계산 (알파와 미래 수익률의 상관계수)
        ic_list = []
        for date in alpha_values.index[:-1]:  # 마지막 날 제외
            alpha_cross_section = alpha_values.loc[date]
            returns_cross_section = returns.loc[date]
            
            # 둘 다 유효한 값이 있는 경우만
            valid = alpha_cross_section.notna() & returns_cross_section.notna()
            if valid.sum() > 10:  # 최소 10개 종목
                ic = alpha_cross_section[valid].corr(returns_cross_section[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) == 0:
            return None
        
        mean_ic = np.mean(ic_list)
        ic_ir = mean_ic / np.std(ic_list) if np.std(ic_list) > 0 else 0
        
        return {
            'ic': mean_ic,
            'ic_ir': ic_ir,
            'ic_std': np.std(ic_list),
            'num_days': len(ic_list)
        }
        
    except Exception as e:
        print(f"   ⚠️  평가 실패: {e}")
        return None

# 메인 함수
def main():
    print("=" * 70)
    print("Alpha-GPT 논문 방식: LLM 기반 알파 생성")
    print("=" * 70)
    print()
    
    # 1. 데이터 로드
    data = load_market_data()
    
    # 2. 트레이딩 아이디어
    trading_idea = """
한국 증시에서 강한 모멘텀을 가진 종목을 찾되, 변동성 대비 수익률이 높은 종목을 선택하고 싶습니다.

핵심 전략:
1. 중기 모멘텀 (20-30일)이 강한 종목
2. 최근 변동성 대비 수익률이 높은 종목 (샤프 비율 개념)
3. 거래량이 평균보다 높아 유동성이 좋은 종목

목표: IC 0.02 이상, IC IR 2.0 이상
"""
    
    print("\n💡 트레이딩 아이디어:")
    print(trading_idea)
    
    # 3. LLM으로 알파 생성
    alphas = generate_alphas_with_llm(trading_idea, num_alphas=10)
    
    print("\n📊 생성된 알파 표현식:")
    for i, alpha in enumerate(alphas, 1):
        print(f"   {i}. {alpha[:100]}...")
    
    # 4. 알파 평가
    print("\n📈 알파 평가 중...")
    
    results = []
    for i, alpha_expr in enumerate(alphas, 1):
        print(f"\n   [{i}/{len(alphas)}] 평가 중...")
        result = evaluate_alpha(alpha_expr, data)
        
        if result:
            results.append({
                'alpha': alpha_expr,
                'ic': result['ic'],
                'ic_ir': result['ic_ir'],
                'ic_std': result['ic_std'],
                'num_days': result['num_days']
            })
            print(f"      IC: {result['ic']:.4f} | IC IR: {result['ic_ir']:.2f}")
        else:
            print(f"      평가 실패")
    
    # 5. 결과 정렬
    results.sort(key=lambda x: x['ic'], reverse=True)
    
    print("\n" + "=" * 70)
    print("✅ 평가 완료!")
    print("=" * 70)
    
    print("\n🏆 상위 5개 알파:")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. IC: {r['ic']:.4f} | IC IR: {r['ic_ir']:.2f}")
        print(f"   {r['alpha']}")
    
    # 6. 최상위 알파 DB 저장
    if results:
        best = results[0]
        print(f"\n💾 최상위 알파 DB 저장...")
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                INSERT INTO alpha_performance
                (alpha_formula, start_date, is_active, total_return, sharpe_ratio, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (alpha_formula, start_date) DO NOTHING
            """, (
                best['alpha'],
                date.today(),
                True,
                0.0,  # 나중에 계산
                best['ic_ir'],
                f"IC: {best['ic']:.4f}, Generated by LLM (Alpha-GPT method)"
            ))
            conn.commit()
            print("✅ DB 저장 완료")
        finally:
            cur.close()
            conn.close()
    
    print("\n🎉 Alpha-GPT 논문 방식 실행 완료!")

if __name__ == "__main__":
    main()

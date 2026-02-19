#!/usr/bin/env python3
"""
한국 시장 특성 기반 알파 가설 테스트
Korean Market-Specific Alpha Hypotheses
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from alpha_gpt_kr.mining.operators import AlphaOperators as ops

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def load_kosdaq200_data():
    """KOSDAQ 200 데이터 + 수급 로드"""
    print("📊 KOSDAQ 200 데이터 로드 중...")
    
    conn = get_db_connection()
    
    query_stocks = """
        SELECT s.id, s.ticker, s.name, s.market_cap
        FROM stocks s
        WHERE s.is_active = true
        AND s.market_cap IS NOT NULL
        AND s.ticker >= '400000'
        ORDER BY s.market_cap DESC
        LIMIT 200
    """
    
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    # 가격 데이터
    query_prices = f"""
        SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= '2019-01-01'
        ORDER BY s.ticker, p.date
    """
    price_df = pd.read_sql(query_prices, conn)
    
    # 수급 데이터
    query_supply = f"""
        SELECT s.ticker, sd.date,
               sd.foreign_net_buy, sd.institution_net_buy,
               sd.individual_net_buy, sd.foreign_ownership
        FROM supply_demand_data sd
        JOIN stocks s ON sd.stock_id = s.id
        WHERE sd.stock_id IN ({stock_id_list})
        AND sd.date >= '2019-01-01'
        ORDER BY s.ticker, sd.date
    """
    supply_df = pd.read_sql(query_supply, conn)
    conn.close()
    
    # Pivot
    close = price_df.pivot(index='date', columns='ticker', values='close')
    high = price_df.pivot(index='date', columns='ticker', values='high')
    low = price_df.pivot(index='date', columns='ticker', values='low')
    open_price = price_df.pivot(index='date', columns='ticker', values='open')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    
    foreign_net = supply_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
    inst_net = supply_df.pivot(index='date', columns='ticker', values='institution_net_buy')
    indiv_net = supply_df.pivot(index='date', columns='ticker', values='individual_net_buy')
    foreign_own = supply_df.pivot(index='date', columns='ticker', values='foreign_ownership')
    
    # 공통 인덱스
    common_idx = close.index.intersection(foreign_net.index)
    
    print(f"  ✅ {len(close.columns)}종목, {len(common_idx)}일")
    
    return {
        'close': close.loc[common_idx],
        'high': high.loc[common_idx],
        'low': low.loc[common_idx],
        'open': open_price.loc[common_idx],
        'volume': volume.loc[common_idx],
        'foreign_net': foreign_net.loc[common_idx],
        'inst_net': inst_net.loc[common_idx],
        'indiv_net': indiv_net.loc[common_idx],
        'foreign_own': foreign_own.loc[common_idx],
    }

# ═══════════════════════════════════════════════════════════════════════
# 한국 시장 알파 가설
# ═══════════════════════════════════════════════════════════════════════

KOREAN_ALPHA_HYPOTHESES = [
    # ── 1. 개미 역행 (Retail Contrarian) ──
    # 가설: 개인 투자자가 많이 파는 종목이 반등한다
    {
        "name": "개미역행_단기",
        "hypothesis": "개인 순매도 급증 → 단기 반등 (개미털기 후 상승)",
        "formula": lambda d: ops.zscore_scale(ops.neg(ops.ts_mean(d['indiv_net'], 5))),
    },
    {
        "name": "개미역행_중기",
        "hypothesis": "개인 20일 순매도 누적 → 중기 반등",
        "formula": lambda d: ops.zscore_scale(ops.neg(ops.ts_sum(d['indiv_net'], 20))),
    },
    
    # ── 2. 외국인 추종 (Foreign Flow Following) ──
    # 가설: 외국인이 꾸준히 사는 종목이 상승
    {
        "name": "외국인추종_모멘텀",
        "hypothesis": "외국인 순매수 가속도 (증가 추세)",
        "formula": lambda d: ops.zscore_scale(ops.ts_delta(ops.ts_mean(d['foreign_net'], 20), 10)),
    },
    {
        "name": "외국인지분_급증",
        "hypothesis": "외국인 지분율 급상승 종목",
        "formula": lambda d: ops.zscore_scale(ops.ts_delta(d['foreign_own'], 20)),
    },
    
    # ── 3. 기관-외국인 동조 (Smart Money Consensus) ──
    # 가설: 기관+외국인 동시 매수 = 강한 신호
    {
        "name": "스마트머니_동조",
        "hypothesis": "기관+외국인 동시 순매수 (개인 역행)",
        "formula": lambda d: ops.add(
            ops.zscore_scale(ops.ts_mean(d['foreign_net'], 10)),
            ops.zscore_scale(ops.ts_mean(d['inst_net'], 10))
        ),
    },
    
    # ── 4. 수급 반전 (Flow Reversal) ──
    # 가설: 외국인 매도세가 꺾이면 반등
    {
        "name": "외국인_반전",
        "hypothesis": "외국인 순매도 → 순매수 전환점",
        "formula": lambda d: ops.zscore_scale(
            ops.sub(ops.ts_mean(d['foreign_net'], 5), ops.ts_mean(d['foreign_net'], 20))
        ),
    },
    
    # ── 5. 거래량 고갈 (Volume Dry-up) ──
    # 가설: 거래량 급감 후 터지는 종목
    {
        "name": "거래량고갈_반등",
        "hypothesis": "거래량 급감 → 에너지 축적 → 급등",
        "formula": lambda d: ops.zscore_scale(ops.neg(ops.div(
            ops.ts_mean(d['volume'], 5),
            ops.ts_mean(d['volume'], 60)
        ))),
    },
    
    # ── 6. 갭 복구 (Gap Recovery) ──
    # 가설: 갭하락 후 복구하는 종목
    {
        "name": "갭하락_복구",
        "hypothesis": "당일 갭하락 but 양봉 마감 = 매수세 유입",
        "formula": lambda d: ops.zscore_scale(ops.ts_mean(
            ops.sub(d['close'] - d['open'], d['open'] - d['close'].shift(1)),
            10
        )),
    },
    
    # ── 7. 변동성 수축 (Volatility Squeeze) ──
    # 가설: 볼린저 밴드 수축 → 폭발 대기
    {
        "name": "볼린저_수축",
        "hypothesis": "가격 변동성 수축 → 다음 움직임 준비",
        "formula": lambda d: ops.zscore_scale(ops.neg(
            ops.div(ops.ts_std(d['close'], 20), ops.ts_mean(d['close'], 20))
        )),
    },
    
    # ── 8. 52주 신저가 반등 (52-Week Low Bounce) ──
    # 가설: 신저가 근처에서 반등
    {
        "name": "신저가_반등",
        "hypothesis": "52주 최저가 대비 위치 (낮을수록 반등 기대)",
        "formula": lambda d: ops.zscore_scale(ops.neg(ops.div(
            d['close'],
            ops.ts_max(d['close'], 240)
        ))),
    },
    
    # ── 9. 월요일 효과 대응 (Monday Effect) ──
    # 가설: 금요일 외국인 순매수 → 월요일 갭업
    {
        "name": "금요일_외국인",
        "hypothesis": "금요일 외국인 순매수 강도",
        "formula": lambda d: ops.zscore_scale(ops.ts_mean(d['foreign_net'], 5)),  # simplified
    },
    
    # ── 10. 아랫꼬리 매집 (Lower Shadow Accumulation) ──
    # 가설: 아랫꼬리가 길면 저점 매수세 유입
    {
        "name": "아랫꼬리_매집",
        "hypothesis": "아랫꼬리 길이 = 저점 매수 강도",
        "formula": lambda d: ops.zscore_scale(ops.ts_mean(
            ops.div(
                np.minimum(d['open'], d['close']) - d['low'],
                d['high'] - d['low'] + 0.0001
            ),
            20
        )),
    },
    
    # ── 11. 기관 선행 (Institutional Lead) ──
    # 가설: 기관이 먼저 사고 외국인이 따라옴
    {
        "name": "기관선행_외국인후행",
        "hypothesis": "기관 순매수 but 외국인 아직 안 산 종목",
        "formula": lambda d: ops.sub(
            ops.zscore_scale(ops.ts_sum(d['inst_net'], 20)),
            ops.zscore_scale(ops.ts_sum(d['foreign_net'], 20))
        ),
    },
    
    # ── 12. 수급 집중도 (Flow Concentration) ──
    # 가설: 특정 세력의 집중 매수
    {
        "name": "외국인_집중매수",
        "hypothesis": "외국인 순매수 / 총거래량 비율",
        "formula": lambda d: ops.zscore_scale(ops.ts_mean(
            ops.div(d['foreign_net'], d['volume'] * d['close'] + 1),
            20
        )),
    },
    
    # ── 13. 저PBR + 외국인 ──
    # 가설: 저평가 + 외국인 관심 = 가치 재발견
    {
        "name": "가치발굴_외국인",
        "hypothesis": "52주 저점 근처 + 외국인 유입",
        "formula": lambda d: ops.add(
            ops.zscore_scale(ops.neg(ops.div(d['close'], ops.ts_max(d['close'], 240)))),
            ops.zscore_scale(ops.ts_delta(d['foreign_own'], 30))
        ),
    },
    
    # ── 14. 이격도 회귀 (Moving Average Reversion) ──
    # 가설: 이격도가 낮으면 평균 회귀
    {
        "name": "이격도_회귀",
        "hypothesis": "20일선 대비 이격도 낮은 종목 반등",
        "formula": lambda d: ops.zscore_scale(ops.neg(ops.div(
            d['close'],
            ops.ts_mean(d['close'], 20)
        ))),
    },
    
    # ── 15. 복합: 저변동성 + 외국인 유입 ──
    {
        "name": "저변동성_외국인",
        "hypothesis": "변동성 낮고 외국인 들어오는 종목",
        "formula": lambda d: ops.add(
            ops.zscore_scale(ops.neg(ops.ts_std(d['close'].pct_change(), 60))),
            ops.zscore_scale(ops.ts_delta(d['foreign_own'], 30))
        ),
    },
]

def calc_ic(alpha_vals, forward_ret, start, end):
    """Calculate IC for a date range"""
    idx_str = pd.to_datetime(alpha_vals.index).strftime('%Y-%m-%d')
    mask = (idx_str >= start) & (idx_str <= end)
    a = alpha_vals.loc[mask]
    r = forward_ret.loc[mask]
    
    ics = []
    for dt in a.index:
        if dt not in r.index:
            continue
        av = a.loc[dt].dropna()
        rv = r.loc[dt].dropna()
        common = av.index.intersection(rv.index)
        if len(common) < 20:
            continue
        ic = av[common].corr(rv[common])
        if not np.isnan(ic):
            ics.append(ic)
    return np.mean(ics) if ics else 0, np.std(ics) if ics else 1

def main():
    data = load_kosdaq200_data()
    
    # 20일 선행 수익률
    forward_ret = data['close'].pct_change(20).shift(-20)
    
    # 4-fold CV
    folds = [
        ('2020-01-01', '2021-06-30', '2021-07-21', '2022-12-31'),
        ('2020-01-01', '2022-06-30', '2022-07-21', '2023-12-31'),
        ('2020-01-01', '2023-06-30', '2023-07-21', '2024-12-31'),
        ('2020-01-01', '2024-06-30', '2024-07-21', '2025-12-31'),
    ]
    
    results = []
    
    print("\n" + "=" * 70)
    print("🇰🇷 한국 시장 알파 가설 테스트")
    print("=" * 70)
    
    for hypo in KOREAN_ALPHA_HYPOTHESES:
        name = hypo['name']
        print(f"\n테스트: {name}")
        print(f"  가설: {hypo['hypothesis']}")
        
        try:
            alpha_vals = hypo['formula'](data)
            
            test_ics = []
            for train_start, train_end, test_start, test_end in folds:
                test_ic, _ = calc_ic(alpha_vals, forward_ret, test_start, test_end)
                test_ics.append(test_ic)
            
            avg_ic = np.mean(test_ics)
            std_ic = np.std(test_ics)
            ir = avg_ic / std_ic if std_ic > 0 else 0
            
            results.append({
                'name': name,
                'hypothesis': hypo['hypothesis'],
                'test_ic': avg_ic,
                'ir': ir,
            })
            print(f"  ✅ Test IC: {avg_ic:+.4f}, IR: {ir:.2f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'name': name,
                'hypothesis': hypo['hypothesis'],
                'test_ic': 0,
                'ir': 0,
            })
    
    # 결과 정렬
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_ic', ascending=False)
    
    print("\n" + "=" * 70)
    print("🏆 한국 시장 알파 순위 (Test IC)")
    print("=" * 70)
    
    for i, row in results_df.iterrows():
        emoji = "🥇" if row['test_ic'] == results_df['test_ic'].max() else "  "
        print(f"{emoji} {row['name']:20s} | IC: {row['test_ic']:+.4f} | IR: {row['ir']:.2f}")
        print(f"     └─ {row['hypothesis']}")
    
    print("\n" + "=" * 70)
    print("📊 기존 최고 대비")
    print("=" * 70)
    print(f"기존 Combined Alpha: IC = 0.1376")
    print(f"신규 최고: {results_df.iloc[0]['name']} IC = {results_df.iloc[0]['test_ic']:.4f}")
    
    # 저장
    results_df.to_csv('/Users/yrbahn/.openclaw/workspace/alpha-gpt-kr/experiments/korean_alpha_results.csv', index=False)
    print("\n결과 저장: experiments/korean_alpha_results.csv")

if __name__ == "__main__":
    main()

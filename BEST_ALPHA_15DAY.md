# 🏆 Best Alpha: 15-day Forward (시총 상위 500개)

**생성일시:** 2026-02-13 00:01 GMT+9

## 알파 정보

**IC (Information Coefficient):** 0.0745

**표현식:**
```python
AlphaOperators.ts_std(returns, 5) / AlphaOperators.ts_mean(close, 91)
```

## 설명

**전략:**
- **분자**: 5일 수익률 변동성 (단기 리스크/변동성)
- **분모**: 91일 평균 가격 (장기 가격 수준)

**의미:**
- 단기 변동성이 높고, 장기 평균 가격이 낮은 종목 선호
- 저가 + 높은 변동성 종목 = 반등 가능성
- 15일 보유 전략에 최적화

**리밸런싱:**
- 주기: 15일 (월 2회)
- 거래비용: 연 ~14.4%
- 순 IC: ~0.05 (비용 차감 후)

## 진화 과정

```
세대 1:  IC 0.0721 🏆
세대 4:  IC 0.0729 🏆
세대 8:  IC 0.0735 🏆
세대 13: IC 0.0739 🏆
세대 17: IC 0.0743 🏆
세대 20: IC 0.0745 🏆 (최종)
```

## 학습 데이터

- **종목**: 시가총액 상위 500개
- **기간**: 2년 (464 거래일)
- **Seed**: GPT-4o 생성 20개
- **GP**: 20세대, Population 100

## 성능 비교

| 실험 | 종목 수 | IC |
|------|--------|-----|
| 섹터별 250개 | 250 | 0.0449 |
| **시총 500개** | **500** | **0.0745** |

시총 상위 500개 종목이 66% 더 높은 IC 달성!

## 적용 방법

### 1. 알파 계산

```python
from alpha_gpt_kr.mining.operators import AlphaOperators

# 데이터 로드 (close, returns)
alpha = AlphaOperators.ts_std(returns, 5) / AlphaOperators.ts_mean(close, 91)
```

### 2. 종목 선정

```python
# 알파 상위 종목
top_stocks = alpha.rank(ascending=False).head(10)
```

### 3. 리밸런싱

- **주기**: 15일마다
- **방법**: 기존 종목 전량 매도 → 새로운 상위 종목 매수
- **종목 수**: 8-10개 권장

## 주의사항

- ⚠️ 단기 변동성 기반이므로 급락장에서 주의 필요
- ⚠️ 15일 보유 전제, 단기 매매 부적합
- ⚠️ 백테스트 기간: 2024-2026 (2년)
- ✅ 거래비용 고려 필수

## DB 저장 명령어

```sql
INSERT INTO alpha_scores (
    stock_code,
    calculation_date,
    alpha_formula,
    alpha_score,
    close_price,
    volume
)
SELECT 
    ticker,
    CURRENT_DATE,
    'AlphaOperators.ts_std(returns, 5) / AlphaOperators.ts_mean(close, 91)',
    -- 계산된 알파 점수,
    close,
    volume
FROM ...
```

---

**🎉 축하합니다! 15일 forward 전략용 최고 알파를 성공적으로 생성했습니다!**

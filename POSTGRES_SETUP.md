# PostgreSQL 데이터베이스 연동 가이드

Alpha-GPT-KR이 PostgreSQL `marketsense` 데이터베이스에서 데이터를 가져오도록 수정되었습니다.

## ✅ 완료된 작업

### 1. 새로운 PostgreSQL 데이터 로더 추가

**파일**: `alpha_gpt_kr/data/postgres_loader.py`

- 192.168.0.248:5432/marketsense 데이터베이스 연결
- stocks, price_data, technical_indicators 테이블 지원
- Panel 형태 데이터 변환 (date × ticker)
- 기술적 지표 포함 옵션

### 2. Core 시스템 수정

**파일**: `alpha_gpt_kr/core.py`

- `KRXDataLoader` → `PostgresDataLoader`로 교체
- `load_data()` 메서드 인터페이스 조정
- 인덱스명(KOSPI200 등) 지원

### 3. 의존성 추가

**파일**: `requirements.txt`

```
psycopg2-binary>=2.9.9  # PostgreSQL 연결
```

## 🚀 사용 방법

### 기본 사용

```python
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader

# 1. 데이터 로더 초기화
loader = PostgresDataLoader(
    host="192.168.0.248",
    port=5432,
    database="marketsense",
    user="yrbahn",
    password="1234"
)

# 2. 최근 데이터 날짜 확인
latest_date = loader.get_latest_date()
print(f"최근 데이터: {latest_date}")

# 3. 종목 리스트 조회 (예: KOSPI200)
# 주의: index_membership 컬럼에 정확한 값이 없으면 빈 리스트 반환
kospi200 = loader.get_universe_by_index("KOSPI200")

# 또는 직접 종목 지정
my_stocks = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, NAVER

# 4. 가격 데이터 로드
data = loader.load_data(
    start_date="2024-01-01",
    end_date="2025-02-11",
    universe=my_stocks,
    include_technical=True  # 기술적 지표 포함
)

# 5. 데이터 확인
print("로드된 데이터:")
for key, df in data.items():
    print(f"  {key}: {df.shape}")

print("\n종가 데이터 샘플:")
print(data['close'].tail())
```

### AlphaGPT와 함께 사용

```python
from alpha_gpt_kr import AlphaGPT

# 1. AlphaGPT 초기화 (내부적으로 PostgresDataLoader 사용)
gpt = AlphaGPT(market="KRX")

# 2. 데이터 로드
data = gpt.load_data(
    universe=["005930", "000660", "035420"],  # 종목 리스트
    start_date="2024-01-01",
    end_date="2025-02-11",
    include_technical=True
)

# 3. 알파 마이닝 (LLM 필요)
# 주의: OpenAI API 키가 설정되어 있어야 함
result = gpt.mine_alpha(
    idea="거래량 급증 + 주가 상승 전략",
    num_seeds=10,
    enhancement_rounds=20
)

print(result.top_alphas)
```

### 백테스팅

```python
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.backtest.engine import BacktestEngine
from alpha_gpt_kr.mining.operators import AlphaOperators as ops

# 1. 데이터 로드
loader = PostgresDataLoader()
data = loader.load_data(
    universe=["005930", "000660"],  # 샘플 종목
    start_date="2024-01-01",
    end_date="2025-02-11"
)

# 2. 알파 생성 (예: 10일 이동평균)
close = data['close']
alpha = ops.ts_mean(close, 10)

# 3. 백테스트
engine = BacktestEngine(
    universe=["005930", "000660"],
    price_data=close,
    return_data=data['returns']
)

result = engine.backtest(
    alpha=alpha,
    alpha_expr="ts_mean(close, 10)",
    quantiles=(0.3, 0.7),
    rebalance_freq='1D'
)

# 4. 결과 확인
print(result.summary())
```

## 📊 데이터베이스 구조

### stocks 테이블
```sql
SELECT id, ticker, name, sector, industry, market_cap
FROM stocks
WHERE is_active = true;
```

### price_data 테이블
```sql
SELECT stock_id, date, open, high, low, close, volume
FROM price_data
WHERE date >= '2024-01-01'
ORDER BY date, stock_id;
```

### technical_indicators 테이블
```sql
SELECT stock_id, date, sma_20, rsi_14, macd, bb_upper
FROM technical_indicators
WHERE date >= '2024-01-01';
```

## 🧪 테스트

### PostgreSQL 연결 테스트

```bash
python3 test_postgres.py
```

**예상 출력**:
```
============================================================
PostgreSQL 데이터 로더 테스트
============================================================

1. 데이터 로더 초기화...
✅ 초기화 성공

2. 최근 데이터 날짜 확인...
✅ 최근 데이터: 2026-02-11

3. KOSPI200 종목 조회...
✅ KOSPI200 종목 수: 0 (또는 실제 종목 수)

4. 가격 데이터 로드...
✅ 로드 완료:
   open: (21, 10) (날짜 × 종목)
   close: (21, 10)
   volume: (21, 10)
   ...

✅ 모든 테스트 통과!
```

### 통합 테스트

```bash
python3 test_alphagpt_postgres.py
```

**예상 출력**:
```
============================================================
Alpha-GPT + PostgreSQL 통합 테스트
============================================================

1. 샘플 종목 선택...
✅ 샘플 종목 20개 선택

2. PostgreSQL에서 데이터 로드...
✅ 데이터 로드 완료

4. 샘플 알파 백테스트...
✅ 백테스트 완료:
   IC: 0.0029
   Sharpe Ratio: -2.01
   연평균 수익률: -43.55%

5. 알파 연산자 테스트...
✅ 연산자 테스트 완료

✅ 모든 테스트 통과!
🎉 Alpha-GPT가 PostgreSQL 데이터로 정상 작동합니다!
```

## ⚙️ 설정

### 데이터베이스 접속 정보 변경

**방법 1**: 코드에서 직접 지정

```python
loader = PostgresDataLoader(
    host="YOUR_HOST",
    port=5432,
    database="YOUR_DB",
    user="YOUR_USER",
    password="YOUR_PASSWORD"
)
```

**방법 2**: `.env` 파일 사용 (향후 추가 가능)

```bash
# .env
POSTGRES_HOST=192.168.0.248
POSTGRES_PORT=5432
POSTGRES_DB=marketsense
POSTGRES_USER=yrbahn
POSTGRES_PASSWORD=1234
```

## 🔧 문제 해결

### 1. 연결 오류

```
psycopg2.OperationalError: could not connect to server
```

**해결**:
- 데이터베이스가 실행 중인지 확인
- 방화벽 설정 확인
- 호스트/포트 정보 확인

### 2. 데이터가 없음

```
가격 데이터: 0 행
```

**해결**:
- 날짜 범위 확인
- 종목 코드가 정확한지 확인
- 데이터베이스에 실제 데이터가 있는지 확인

```sql
-- 데이터 존재 확인
SELECT COUNT(*), MIN(date), MAX(date)
FROM price_data;
```

### 3. KOSPI200 종목이 0개

```
KOSPI200 종목 수: 0
```

**원인**: `stocks.index_membership` 컬럼에 "KOSPI200" 문자열이 정확히 없음

**해결**:
- 직접 종목 리스트 지정
- 또는 index_membership 컬럼 값 확인

```sql
-- index_membership 값 확인
SELECT DISTINCT index_membership
FROM stocks
WHERE index_membership IS NOT NULL;
```

## 📚 다음 단계

1. **LLM 기능 사용** (OpenAI API 키 필요):
   ```bash
   echo "OPENAI_API_KEY=your-key-here" > .env
   ```

2. **실제 알파 마이닝**:
   ```python
   result = gpt.mine_alpha("모멘텀 전략", num_seeds=10)
   ```

3. **자동 트레이딩 연동** (선택):
   - 키움증권 API
   - 한국투자증권 API

## 📝 주의사항

1. **종목 코드 포맷**:
   - 6자리 문자열: "005930" (삼성전자)
   - 자동으로 0-padding 처리됨

2. **날짜 형식**:
   - "YYYY-MM-DD" 형식 사용
   - 예: "2024-01-01"

3. **성능**:
   - 대량 데이터 로드 시 시간 소요
   - 캐싱 권장 (향후 추가 예정)

4. **거래 비용**:
   - 백테스트 기본값: 수수료 0.15%, 슬리피지 0.1%
   - 실제 거래와 차이 발생 가능

## 🆘 도움말

문제가 발생하면:
1. `test_postgres.py` 실행해서 연결 확인
2. 데이터베이스 로그 확인
3. PostgreSQL 쿼리로 직접 데이터 확인

---

**마지막 업데이트**: 2026-02-12  
**데이터베이스**: marketsense @ 192.168.0.248:5432  
**테스트 상태**: ✅ 모든 테스트 통과

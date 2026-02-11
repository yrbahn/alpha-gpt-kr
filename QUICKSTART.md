# Quick Start Guide

## 설치

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/alpha-gpt-kr.git
cd alpha-gpt-kr
```

### 2. 가상 환경 생성 (권장)
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt

# 또는 개발 모드로 설치
pip install -e .
```

### 4. 환경 변수 설정
```bash
cp .env.example .env
```

`.env` 파일을 열어서 API 키 설정:
```
OPENAI_API_KEY=your_key_here
# 또는
ANTHROPIC_API_KEY=your_key_here
```

## 빠른 테스트

### 기본 기능 테스트
```bash
python3 experiments/paper_replication.py
```

**예상 출력:**
```
============================================================
Alpha-GPT Paper Replication Tests
============================================================
...
✅ ALL TESTS PASSED
```

## 기본 사용법

### 예제 1: 간단한 알파 생성

```python
from alpha_gpt_kr import AlphaGPT

# AlphaGPT 초기화
gpt = AlphaGPT(
    market="KRX",
    llm_provider="openai",
    model="gpt-4-turbo-preview"
)

# 테스트 데이터 로드 (일부 종목만)
test_stocks = ["005930", "000660", "035420"]  # 삼성전자, SK하이닉스, NAVER
gpt.load_data(
    universe=test_stocks,
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# 알파 마이닝
idea = "거래량이 증가하면서 주가도 상승하는 모멘텀 종목"
result = gpt.mine_alpha(idea, num_seeds=3, enhancement_rounds=5)

# 결과 확인
for expr, backtest in result.top_alphas:
    print(f"Alpha: {expr}")
    print(f"IC: {backtest.ic:.4f}")
    print(f"Sharpe: {backtest.sharpe_ratio:.2f}")
    print()
```

### 예제 2: 연산자 직접 사용

```python
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from alpha_gpt_kr.data import load_krx_data
import pandas as pd

# 데이터 로드
data = load_krx_data(
    tickers=["005930", "000660"],
    start_date="2023-01-01",
    end_date="2024-01-01",
    fields=["close", "volume"]
)

close = data['close']
volume = data['volume']

# 알파 계산
# 예: 20일 모멘텀
momentum = ops.ts_delta(close, 20) / close.shift(20)

# 예: 거래량-가격 상관관계
corr = ops.ts_corr(volume, close, 20)

# 예: 산업 중립화 (sector_groups는 별도 로드 필요)
# neutral_alpha = ops.grouped_demean(momentum, sector_groups)

print(momentum.head())
print(corr.head())
```

### 예제 3: 백테스팅

```python
from alpha_gpt_kr.backtest import BacktestEngine
from alpha_gpt_kr.mining.operators import ops
import pandas as pd

# 데이터 준비 (위와 동일)
# ...

# 알파 정의
alpha = ops.ts_rank(ops.ts_delta(close, 20), 10)

# 백테스트
engine = BacktestEngine(
    universe=["005930", "000660"],
    price_data=close
)

result = engine.backtest(alpha, alpha_expr="momentum_rank")

# 결과 출력
print(result.summary())
```

## 실험 실행

### 한국 증시 벤치마크
```bash
python3 experiments/krx_benchmark.py --start 2020-01-01 --end 2024-12-31
```

**참고:** 실제 데이터를 사용하므로 시간이 걸릴 수 있습니다.

### 논문 재현 실험
```bash
python3 experiments/paper_replication.py
```

## 문제 해결

### FinanceDataReader 설치 오류
```bash
pip install finance-datareader
```

### LLM API 오류
- `.env` 파일에 API 키가 올바르게 설정되었는지 확인
- API 키의 유효성 확인
- 네트워크 연결 확인

### 데이터 로드 실패
- FinanceDataReader가 KRX 서버에 접근 가능한지 확인
- 인터넷 연결 확인
- 종목 코드가 올바른지 확인

## 다음 단계

1. **아키텍처 이해**: `ARCHITECTURE.md` 참조
2. **상세 문서**: `docs/` 폴더 확인
3. **고급 예제**: `examples/` 폴더 탐색
4. **커스터마이징**: 연산자 추가, 새로운 전략 개발

## 지원

- Issues: GitHub Issues
- 논문: [arXiv:2308.00016](https://arxiv.org/abs/2308.00016)
- Documentation: `docs/` 폴더

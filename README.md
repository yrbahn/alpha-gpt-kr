# Alpha-GPT-KR: 한국 증시용 AI 기반 알파 마이닝 시스템

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

arXiv 2308.00016 "Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment" 논문의 한국 증시 구현체

## 📖 논문 개요

Alpha-GPT는 대규모 언어모델(LLM)을 활용하여 퀀트 투자자와 AI가 협력적으로 알파(trading signals)를 발굴하는 새로운 패러다임입니다.

### 핵심 특징
- **Human-AI Interactive Mining**: 자연어로 트레이딩 아이디어 입력
- **Agentic Workflow**: Ideation → Implementation → Review
- **Genetic Programming**: 초기 알파를 진화적으로 개선
- **한국 증시 최적화**: KRX 데이터, 한국어 지원

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────┐
│           User Interface (WebUI/CLI)            │
├─────────────────────────────────────────────────┤
│              AlphaBot (LLM Agent)               │
│  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Trading Idea │  │  Quant Developer       │  │
│  │  Polisher    │  │  (Alpha Generator)     │  │
│  └──────────────┘  └────────────────────────┘  │
│  ┌──────────────────────────────────────────┐  │
│  │      Analyst (Result Interpreter)        │  │
│  └──────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│       Algorithmic Alpha Mining Engine           │
│  • Genetic Programming                          │
│  • Alpha Search Enhancement                     │
│  • Backtesting & Evaluation                     │
├─────────────────────────────────────────────────┤
│      Korean Market Data Layer                   │
│  • KRX Stock Data (FinanceDataReader)          │
│  • OHLCV + Volume-weighted Data                 │
│  • Industry/Sector Classification               │
└─────────────────────────────────────────────────┘
```

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/alpha-gpt-kr.git
cd alpha-gpt-kr

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 등을 설정
```

### 기본 사용법

```python
from alpha_gpt_kr import AlphaGPT

# 시스템 초기화
gpt = AlphaGPT(
    market="KRX",
    llm_provider="openai",
    model="gpt-4"
)

# Interactive Mode: 트레이딩 아이디어 입력
idea = """
거래량이 급증하면서 주가가 상승하는 종목을 찾고 싶습니다.
20일 이동평균 대비 거래량이 2배 이상이고,
종가가 전일 대비 3% 이상 상승한 경우를 포착해주세요.
"""

# 알파 생성 및 최적화
results = gpt.mine_alpha(
    idea=idea,
    num_seeds=10,
    enhancement_rounds=20,
    mode="interactive"
)

# 결과 확인
print(results.top_alphas)
print(f"Best IC: {results.best_ic:.4f}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

## 📊 주요 기능

### 1. Interactive Alpha Mining
```python
# 자연어로 아이디어 제시
alpha = gpt.chat("모멘텀과 밸류 팩터를 결합한 전략을 만들어줘")

# 생성된 알파 검토 및 피드백
feedback = "산업별로 중립화해서 다시 만들어줘"
improved_alpha = gpt.refine(alpha, feedback)
```

### 2. Autonomous Mode
```python
# 대규모 데이터베이스 자동 탐색
results = gpt.mine_alpha(
    mode="autonomous",
    explore_fields=["price-volume", "fundamental", "sentiment"],
    max_iterations=100
)
```

### 3. 백테스팅 및 평가
```python
# 생성된 알파 백테스트
backtest = gpt.backtest(
    alpha_expr="ts_corr(volume, close, 20)",
    start_date="2020-01-01",
    end_date="2024-12-31",
    universe="KOSPI200"
)

print(f"연평균 수익률: {backtest.annual_return:.2%}")
print(f"Information Coefficient: {backtest.ic:.4f}")
print(f"Sharpe Ratio: {backtest.sharpe:.2f}")
```

## 📁 프로젝트 구조

```
alpha-gpt-kr/
├── alpha_gpt_kr/              # 메인 패키지
│   ├── agents/                # LLM 에이전트
│   │   ├── trading_idea_polisher.py
│   │   ├── quant_developer.py
│   │   └── analyst.py
│   ├── mining/                # 알파 마이닝 엔진
│   │   ├── genetic_programming.py
│   │   ├── alpha_search.py
│   │   └── operators.py
│   ├── data/                  # 한국 증시 데이터
│   │   ├── krx_loader.py
│   │   └── data_processor.py
│   ├── backtest/              # 백테스팅
│   │   ├── engine.py
│   │   └── metrics.py
│   ├── knowledge/             # 지식 베이스
│   │   ├── alpha_library.py
│   │   └── embeddings.py
│   └── core.py                # 메인 AlphaGPT 클래스
├── experiments/               # 실험 및 검증
│   ├── paper_replication.py   # 논문 재현 실험
│   ├── krx_benchmark.py       # 한국 증시 벤치마크
│   └── case_studies.ipynb
├── data/                      # 데이터 저장소
│   ├── raw/                   # 원본 데이터
│   ├── processed/             # 전처리된 데이터
│   └── cache/                 # 캐시
├── configs/                   # 설정 파일
│   ├── operators.yaml         # 연산자 정의
│   ├── prompts/               # 프롬프트 템플릿
│   └── default.yaml
├── tests/                     # 테스트
├── docs/                      # 문서
├── requirements.txt
├── setup.py
└── README.md
```

## 🔬 논문 구현 상세

### 연산자 구현 (Operators)

논문의 Table 1에 정의된 모든 연산자 구현:

**Time-series operators:**
- `ts_corr`, `ts_cov`, `ts_mean`, `ts_std`, `ts_rank`, `ts_delta`, `ts_ema`, etc.

**Cross-sectional operators:**
- `zscore_scale`, `winsorize_scale`, `normed_rank`, `cwise_max`, `cwise_min`

**Group-wise operators:**
- `grouped_demean`, `grouped_zscore_scale`, `grouped_max`, etc.

**Element-wise operators:**
- `relu`, `abs`, `log`, `sign`, `pow`, `add`, `minus`, `div`, etc.

### 평가 지표 (Evaluation Metrics)

- **Information Coefficient (IC)**: 알파와 미래 수익률 간 상관관계
- **Sharpe Ratio**: 위험 대비 수익률
- **Turnover**: 포트폴리오 회전율
- **Maximum Drawdown (MDD)**: 최대 낙폭

## 📈 실험 결과

### 논문 재현 실험
```bash
python experiments/paper_replication.py
```

### 한국 증시 벤치마크
```bash
python experiments/krx_benchmark.py --start 2020-01-01 --end 2024-12-31
```

## 🛠️ 개발 로드맵

- [x] 논문 분석 및 아키텍처 설계
- [x] 핵심 연산자 구현
- [x] 한국 증시 데이터 로더
- [ ] LLM 에이전트 (GPT-4/Claude)
- [ ] Genetic Programming 엔진
- [ ] 백테스팅 엔진
- [ ] WebUI
- [ ] 실험 검증
- [ ] 문서화

## 📚 참고 문헌

```bibtex
@article{wang2023alphagpt,
  title={Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment},
  author={Wang, Saizhuo and Yuan, Hang and Zhou, Leon and Ni, Lionel M. and Shum, Heung-Yeung and Guo, Jian},
  journal={arXiv preprint arXiv:2308.00016},
  year={2023}
}
```

## 📄 라이센스

MIT License

## 🤝 기여

이슈 및 PR 환영합니다!

## 📧 연락처

프로젝트 관련 문의: [your-email@example.com]

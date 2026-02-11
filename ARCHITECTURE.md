# Alpha-GPT-KR Architecture

## 논문 기반 시스템 아키텍처

본 프로젝트는 arXiv 2308.00016 "Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment" 논문의 아키텍처를 충실히 구현합니다.

## 1. Agentic Workflow (Figure 2)

```
┌─────────────────────────────────────────────────────────┐
│                    IDEATION STAGE                       │
│                                                         │
│  User Trading Idea                                      │
│         ↓                                               │
│  [Trading Idea Polisher]                                │
│    - Natural language processing                        │
│    - Idea structuring                                   │
│    - Data field identification                          │
│    - Knowledge base retrieval                           │
│         ↓                                               │
│  Polished & Structured Idea                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                 IMPLEMENTATION STAGE                    │
│                                                         │
│  [Quant Developer Agent]                                │
│    - Seed alpha generation (LLM)                        │
│    - Alpha expression synthesis                         │
│    - Operator combination                               │
│         ↓                                               │
│  Seed Alphas → [Alpha Database]                         │
│         ↓                                               │
│  [Alpha Compute Framework]                              │
│    - Genetic Programming                                │
│    - Crossover & Mutation                               │
│    - Fitness evaluation (IC)                            │
│    - Population evolution                               │
│         ↓                                               │
│  Enhanced Alphas                                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                    REVIEW STAGE                         │
│                                                         │
│  [Backtesting Engine]                                   │
│    - Historical simulation                              │
│    - IC calculation                                     │
│    - Performance metrics                                │
│         ↓                                               │
│  Backtest Results                                       │
│         ↓                                               │
│  [Analyst Agent]                                        │
│    - Result interpretation                              │
│    - Natural language summary                           │
│    - Strengths & weaknesses                             │
│    - Improvement suggestions                            │
│         ↓                                               │
│  Analysis Report → User Feedback → Next Iteration       │
└─────────────────────────────────────────────────────────┘
```

## 2. System Architecture (Figure 6 기반)

### Layer 1: User Interface
- **WebUI** (향후 구현 예정)
  - Dialog Box: 자연어 대화
  - Mining Session Manager: 세션 관리
  - Alpha Mining Dashboard: 결과 시각화

- **CLI** (현재 구현)
  - Python API
  - Command-line interface

### Layer 2: LLM Agent (AlphaBot)
```python
alpha_gpt_kr/agents/
├── trading_idea_polisher.py  # RAG + Prompt Engineering
├── quant_developer.py         # Code Generation
└── analyst.py                 # Result Interpretation
```

**핵심 기능:**
- Retrieval-Augmented Generation (RAG)
- Structured output parsing
- Domain knowledge compilation/decompilation

### Layer 3: Algorithmic Alpha Mining
```python
alpha_gpt_kr/mining/
├── operators.py              # Table 1 연산자
├── genetic_programming.py    # GP 진화 알고리즘
└── alpha_search.py           # 검색 전략 (향후)
```

**구성 요소:**
- **Alpha Search Enhancement**
  - Genetic Programming
  - Crossover: 부분 표현식 교환
  - Mutation: 연산자/파라미터 변경
  - Selection: IC 기반 토너먼트 선택

- **Evaluation & Backtesting**
  - IC (Information Coefficient)
  - Sharpe Ratio
  - Turnover analysis

### Layer 4: Data & Computation
```python
alpha_gpt_kr/data/
└── krx_loader.py  # 한국 증시 데이터
```

**데이터 소스:**
- FinanceDataReader (KRX)
- pykrx (보조)
- OHLCV, VWAP, Volume
- Sector classification

## 3. Two Modes of Operation (Section 3)

### Interactive Mode
```
User → Idea → AlphaGPT → Alphas → User Review → Refinement → ...
```

**특징:**
- 인간 전문성 + AI 계산 능력
- 반복적 개선
- 사용자 피드백 반영

### Autonomous Mode (향후 확장)
```
Database Exploration → Hierarchical RAG → Alpha Generation → ...
```

**계층적 RAG 전략 (Figure 3):**
1. Alpha Database 분석
2. High-level Categories 탐색
3. Second-level Categories 검색
4. Specific Data Fields 활용

## 4. Key Components

### 4.1 Operators (Table 1 완전 구현)

**Time-series (19개)**
- ts_delta, ts_mean, ts_std, ts_corr, ts_ema, ts_rank, ...

**Cross-sectional (3개)**
- zscore_scale, winsorize_scale, normed_rank

**Group-wise (7개)**
- grouped_demean, grouped_zscore_scale, ...

**Element-wise (14개)**
- abs, log, sign, add, minus, div, ...

### 4.2 Genetic Programming

**알고리즘:**
```python
1. Initialize population from seed alphas
2. For each generation:
   a. Evaluate fitness (IC)
   b. Select parents (tournament)
   c. Crossover (expression mixing)
   d. Mutation (operator/parameter change)
   e. Elitism (preserve top individuals)
3. Return best alphas
```

**파라미터 (논문 기반):**
- Population size: 100
- Generations: 20
- Crossover prob: 0.6
- Mutation prob: 0.3

### 4.3 Backtesting

**평가 지표:**
- **IC**: Spearman correlation (alpha vs. forward returns)
- **IR**: IC / IC_std
- **Sharpe Ratio**: Annualized return / risk
- **MDD**: Maximum Drawdown
- **Turnover**: Portfolio rebalancing frequency

## 5. 한국 증시 특화

### 5.1 데이터 적응
- KRX 거래일 기준
- 한국 산업 분류 (KOSPI/KOSDAQ)
- 거래 수수료: 0.15% (편도)
- 슬리피지: 0.1%

### 5.2 언어 적응
- 한국어 프롬프트
- 한국어 분석 리포트
- 한국 증시 용어

### 5.3 규제 고려
- 공매도 제한 고려
- 가격 제한폭 (상하한가)
- 시장 특성 반영

## 6. 확장 가능성

### 향후 구현 예정
- [ ] WebUI (Streamlit/Gradio)
- [ ] Autonomous mode (hierarchical RAG)
- [ ] Multi-factor combination
- [ ] Real-time deployment
- [ ] Risk management module
- [ ] Portfolio optimization

## 7. 실험 재현

### 논문 실험 (Section 5)

**5.1 Efficiency Improvement**
- Translation Consistency
- Human-AI Iterative Refinement

**5.2 Search Enhancement**
- IC convergence curve (Figure 4)

**5.3 Stronger Alphas**
- Benchmark comparison
- WorldQuant IQC 성과 재현

### 한국 증시 벤치마크
```bash
python experiments/krx_benchmark.py --start 2020-01-01 --end 2024-12-31
```

## 참고 자료

- 논문: [arXiv:2308.00016](https://arxiv.org/abs/2308.00016)
- 프로젝트: `/Users/yrbahn/.openclaw/workspace/alpha-gpt-kr`
- 논문 PDF: `alpha_gpt_paper.pdf`

# Alpha-GPT-KR: DB 기반 트레이딩 시스템

## 개요

알파 계산 결과를 PostgreSQL DB에 저장하고, 다음 날 아침 저장된 데이터로 매수하는 시스템입니다.

### 워크플로우

```
[오후 5시] 알파 계산 → DB 저장
     ↓
[다음날 오전 9시] DB에서 읽기 → 매수 실행
     ↓
[언제든지] 대시보드로 현황 확인
```

## 1. 초기 설정

### DB 스키마 생성

```bash
cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr
python setup_db.py
```

생성되는 테이블:
- `alpha_scores`: 매일 계산된 알파 스코어
- `trading_signals`: 매수/매도 신호
- `portfolio_history`: 포트폴리오 이력
- `account_history`: 계좌 상태 이력
- `alpha_performance`: 알파 성과 추적

## 2. 매일 실행

### 2.1 오후: 알파 계산 및 저장

```bash
# 상위 500개 종목 알파 계산 → DB 저장
python calculate_and_save_alpha.py
```

**출력 예시:**
```
📊 Loading data for top 500 stocks...
📈 Calculating alpha: ops.ts_delta(close, 26)
✅ Saved 477 alpha scores to database

📊 Top 10 Alpha Scores:
rank  stock_code  stock_name    alpha_score  close_price
   1  005930      삼성전자         0.025431    72000
   2  000660      SK하이닉스       0.023891    145000
   ...
```

### 2.2 다음날 오전: DB에서 매수

```bash
# DRY RUN (시뮬레이션만)
python trade_from_db.py --top-n 15 --amount 5000000 --dry-run

# 실제 매수 (신중!)
python trade_from_db.py --top-n 15 --amount 5000000
```

**옵션:**
- `--top-n`: 상위 N개 종목 선택 (기본: 15)
- `--amount`: 총 투자 금액 (기본: 5,000,000원)
- `--dry-run`: 시뮬레이션 모드 (실제 주문 없음)

**출력 예시:**
```
📅 Latest alpha calculation date: 2026-02-12

📊 Top 15 stocks from DB:
rank  stock_code  stock_name    alpha_score  close_price
   1  005930      삼성전자         0.025431    72000
   2  000660      SK하이닉스       0.023891    145000
   ...

💰 Investment Plan:
Total amount: 5,000,000원
Per stock: 333,333원
Number of stocks: 15
Mode: REAL TRADING

⚠️  Real trading mode! Continue? (yes/no):
```

### 2.3 대시보드 생성

```bash
# HTML 대시보드 생성
python generate_dashboard.py
```

브라우저에서 열기:
```
file:///Users/yrbahn/.openclaw/workspace/alpha-gpt-kr/dashboard.html
```

**대시보드 내용:**
- 현재 사용 중인 알파 공식
- 계좌 현황 (총 자산, 현금, 수익률)
- 포트폴리오 상세 (종목별 손익)
- 계좌 가치 추이 차트
- 알파 스코어 분포 차트
- 최근 매매 신호 내역

## 3. 자동화 (Cron)

### 3.1 매일 오후 5시: 알파 계산

```bash
# crontab -e
0 17 * * 1-5 cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr && /usr/local/bin/python calculate_and_save_alpha.py >> logs/alpha_calc.log 2>&1
```

### 3.2 매일 오전 8시 50분: 매수 준비

```bash
# 장 시작 10분 전에 DB 확인 및 시뮬레이션
50 8 * * 1-5 cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr && /usr/local/bin/python trade_from_db.py --dry-run >> logs/trade_check.log 2>&1
```

### 3.3 매일 저녁 6시: 대시보드 업데이트

```bash
0 18 * * 1-5 cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr && /usr/local/bin/python generate_dashboard.py >> logs/dashboard.log 2>&1
```

## 4. DB 직접 조회

### 최신 알파 스코어 확인

```sql
SELECT * FROM latest_alpha_scores LIMIT 10;
```

### 미실행 매매 신호

```sql
SELECT * FROM pending_signals;
```

### 현재 포트폴리오

```sql
SELECT * FROM current_portfolio;
```

### 계좌 이력 (최근 30일)

```sql
SELECT 
    record_date,
    total_balance,
    cash_balance,
    stock_value,
    total_profit_loss_pct
FROM account_history
ORDER BY record_date DESC
LIMIT 30;
```

## 5. 포트폴리오 상태 기록

### 수동 기록 (필요시)

```python
from alpha_gpt_kr.trading.kis_api import KISAPI
import psycopg2
from datetime import date

# KIS API로 잔고 조회
api = KISAPI(...)
balance = api.get_balance()

# DB에 저장
conn = psycopg2.connect(...)
cur = conn.cursor()

# portfolio_history 저장
for stock in balance['stocks']:
    cur.execute("""
        INSERT INTO portfolio_history
        (record_date, stock_code, stock_name, quantity, avg_price, current_price, 
         market_value, profit_loss, profit_loss_pct, weight)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (record_date, stock_code) DO UPDATE SET
            current_price = EXCLUDED.current_price,
            market_value = EXCLUDED.market_value,
            profit_loss = EXCLUDED.profit_loss,
            profit_loss_pct = EXCLUDED.profit_loss_pct
    """, (
        date.today(),
        stock['code'],
        stock['name'],
        stock['qty'],
        stock['avg_price'],
        stock['current_price'],
        stock['value'],
        stock['pl'],
        stock['pl_pct'],
        stock['weight']
    ))

# account_history 저장
cur.execute("""
    INSERT INTO account_history
    (record_date, total_balance, cash_balance, stock_value, 
     total_profit_loss, total_profit_loss_pct, num_holdings, alpha_formula)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (record_date) DO UPDATE SET
        total_balance = EXCLUDED.total_balance,
        cash_balance = EXCLUDED.cash_balance,
        stock_value = EXCLUDED.stock_value,
        total_profit_loss = EXCLUDED.total_profit_loss,
        total_profit_loss_pct = EXCLUDED.total_profit_loss_pct,
        num_holdings = EXCLUDED.num_holdings
""", (
    date.today(),
    balance['total'],
    balance['cash'],
    balance['stock_value'],
    balance['total_pl'],
    balance['total_pl_pct'],
    len(balance['stocks']),
    'ops.ts_delta(close, 26)'
))

conn.commit()
```

## 6. 주의사항

### 리스크 관리
- 항상 `--dry-run`으로 먼저 테스트
- 실제 매수 전 잔고 확인 (`check_balance.py`)
- 투자 금액은 가용 현금 이내로 설정
- Stop-loss (-5%), Take-profit (+10%) 고려

### 데이터 검증
- 알파 계산 전 DB 데이터 최신 여부 확인
- 매수 전 가격 데이터가 당일 것인지 확인
- 주말/공휴일에는 실행 안 됨 (cron 1-5)

### 백업
- DB 정기 백업 권장
- 매매 로그 보관 (`logs/` 디렉터리)

## 7. 문제 해결

### "No alpha scores found"
```bash
# 알파 계산 다시 실행
python calculate_and_save_alpha.py
```

### "Database connection failed"
```bash
# .env 파일 확인
cat .env | grep DB_

# PostgreSQL 연결 테스트
python test_postgres.py
```

### "No data loaded"
```bash
# price_data 테이블 확인
psql -h 192.168.0.248 -U yrbahn -d marketsense -c "SELECT COUNT(*) FROM price_data;"
```

## 8. 파일 구조

```
alpha-gpt-kr/
├── db_schema.sql              # DB 스키마 정의
├── setup_db.py                # DB 초기화
├── calculate_and_save_alpha.py  # 알파 계산 및 저장
├── trade_from_db.py           # DB에서 읽어 매수
├── generate_dashboard.py      # 대시보드 생성
├── dashboard.html             # 생성된 대시보드 (브라우저로 열기)
└── logs/                      # 실행 로그
    ├── alpha_calc.log
    ├── trade_check.log
    └── dashboard.log
```

## 9. 다음 단계

1. **포트폴리오 리밸런싱**: 5일마다 재조정
2. **리스크 모니터링**: Stop-loss/Take-profit 자동화
3. **알파 진화**: GP를 통한 지속적 개선
4. **성과 분석**: 백테스트 vs 실제 성과 비교

---

**현재 설정:**
- 알파: `ops.ts_delta(close, 26)` (26일 모멘텀)
- 성과: IC 0.0045, Sharpe 0.57, Return +21% (2yr backtest)
- 계좌: 44009082-01 (KIS 실계좌)

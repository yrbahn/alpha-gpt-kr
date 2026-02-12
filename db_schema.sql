-- Alpha-GPT-KR Trading Database Schema

-- 알파 점수 저장 테이블
CREATE TABLE IF NOT EXISTS alpha_scores (
    id SERIAL PRIMARY KEY,
    calculation_date DATE NOT NULL,
    calculation_time TIMESTAMP NOT NULL DEFAULT NOW(),
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(100),
    alpha_formula TEXT NOT NULL,
    alpha_score FLOAT NOT NULL,
    rank INTEGER,
    market_cap BIGINT,
    close_price FLOAT,
    volume BIGINT,
    UNIQUE(calculation_date, stock_code, alpha_formula)
);

-- 매매 신호 테이블
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    signal_date DATE NOT NULL,
    signal_time TIMESTAMP NOT NULL DEFAULT NOW(),
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(100),
    signal_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    alpha_score FLOAT,
    rank INTEGER,
    target_weight FLOAT,
    reason TEXT,
    executed BOOLEAN DEFAULT FALSE,
    execution_time TIMESTAMP,
    execution_price FLOAT,
    execution_quantity INTEGER,
    UNIQUE(signal_date, stock_code)
);

-- 포트폴리오 이력 테이블
CREATE TABLE IF NOT EXISTS trading_portfolio (
    id SERIAL PRIMARY KEY,
    record_date DATE NOT NULL,
    record_time TIMESTAMP NOT NULL DEFAULT NOW(),
    stock_code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(100),
    quantity INTEGER NOT NULL,
    avg_price FLOAT NOT NULL,
    current_price FLOAT,
    market_value BIGINT,
    profit_loss BIGINT,
    profit_loss_pct FLOAT,
    weight FLOAT,
    UNIQUE(record_date, stock_code)
);

-- 계좌 상태 이력
CREATE TABLE IF NOT EXISTS trading_account (
    id SERIAL PRIMARY KEY,
    record_date DATE NOT NULL,
    record_time TIMESTAMP NOT NULL DEFAULT NOW(),
    total_balance BIGINT NOT NULL,
    cash_balance BIGINT NOT NULL,
    stock_value BIGINT NOT NULL,
    total_profit_loss BIGINT,
    total_profit_loss_pct FLOAT,
    num_holdings INTEGER,
    alpha_formula TEXT,
    notes TEXT,
    UNIQUE(record_date)
);

-- 알파 성과 추적
CREATE TABLE IF NOT EXISTS alpha_performance (
    id SERIAL PRIMARY KEY,
    alpha_formula TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    total_return FLOAT,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    win_rate FLOAT,
    num_trades INTEGER,
    avg_holding_days FLOAT,
    notes TEXT,
    UNIQUE(alpha_formula, start_date)
);

-- 기존 뷰 삭제 (충돌 방지)
DROP VIEW IF EXISTS latest_alpha_scores CASCADE;
DROP VIEW IF EXISTS pending_signals CASCADE;
DROP VIEW IF EXISTS current_portfolio CASCADE;

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_alpha_scores_date ON alpha_scores(calculation_date DESC);
CREATE INDEX IF NOT EXISTS idx_alpha_scores_rank ON alpha_scores(calculation_date, rank);
CREATE INDEX IF NOT EXISTS idx_trading_signals_date ON trading_signals(signal_date DESC);
CREATE INDEX IF NOT EXISTS idx_trading_portfolio_date ON trading_portfolio(record_date DESC);
CREATE INDEX IF NOT EXISTS idx_trading_account_date ON trading_account(record_date DESC);

-- 뷰: 최신 알파 스코어 (오늘 또는 가장 최근)
CREATE VIEW latest_alpha_scores AS
SELECT *
FROM alpha_scores
WHERE calculation_date = (SELECT MAX(calculation_date) FROM alpha_scores)
ORDER BY rank;

-- 뷰: 미실행 매매 신호
CREATE VIEW pending_signals AS
SELECT *
FROM trading_signals
WHERE executed = FALSE
ORDER BY signal_date DESC, rank;

-- 뷰: 현재 포트폴리오
CREATE VIEW current_portfolio AS
SELECT *
FROM trading_portfolio
WHERE record_date = (SELECT MAX(record_date) FROM trading_portfolio)
ORDER BY weight DESC;

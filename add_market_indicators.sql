-- 시장 지수 및 환율 데이터 테이블 생성

CREATE TABLE IF NOT EXISTS market_indicators (
    date DATE PRIMARY KEY,
    kospi_close DECIMAL(10, 2),
    kospi_volume BIGINT,
    kosdaq_close DECIMAL(10, 2),
    kosdaq_volume BIGINT,
    usd_krw DECIMAL(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_market_date ON market_indicators(date);

-- 시장 지표 계산 뷰
CREATE OR REPLACE VIEW market_regime AS
SELECT 
    date,
    kospi_close,
    kospi_close / LAG(kospi_close, 20) OVER (ORDER BY date) - 1 as kospi_return_20d,
    kospi_close / LAG(kospi_close, 60) OVER (ORDER BY date) - 1 as kospi_return_60d,
    kosdaq_close / kospi_close as kosdaq_kospi_ratio,
    usd_krw / LAG(usd_krw, 20) OVER (ORDER BY date) - 1 as usd_krw_change_20d
FROM market_indicators
ORDER BY date;

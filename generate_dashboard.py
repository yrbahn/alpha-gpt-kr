#!/usr/bin/env python3
"""
Alpha-GPT-KR 대시보드 생성
알파 사용 내역, 포트폴리오, 성과를 시각화
"""
import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import json
from dotenv import load_dotenv
import psycopg2

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
load_dotenv()

def get_db_connection():
    """PostgreSQL 연결"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def get_alpha_history():
    """알파 계산 내역"""
    conn = get_db_connection()
    query = """
        SELECT 
            calculation_date,
            alpha_formula,
            COUNT(*) as num_stocks,
            AVG(alpha_score) as avg_score,
            MAX(alpha_score) as max_score,
            MIN(alpha_score) as min_score
        FROM alpha_scores
        GROUP BY calculation_date, alpha_formula
        ORDER BY calculation_date DESC
        LIMIT 30
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_trading_signals():
    """매매 신호 내역"""
    conn = get_db_connection()
    query = """
        SELECT 
            signal_date,
            stock_code,
            stock_name,
            signal_type,
            alpha_score,
            rank,
            executed,
            execution_price,
            execution_quantity
        FROM trading_signals
        ORDER BY signal_date DESC, rank
        LIMIT 100
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_latest_portfolio():
    """최신 포트폴리오"""
    conn = get_db_connection()
    query = """
        SELECT *
        FROM current_portfolio
        ORDER BY weight DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_account_history():
    """계좌 이력"""
    conn = get_db_connection()
    query = """
        SELECT *
        FROM trading_account
        ORDER BY record_date DESC
        LIMIT 30
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def generate_html_dashboard():
    """HTML 대시보드 생성"""
    
    # 데이터 로드
    df_alpha = get_alpha_history()
    df_signals = get_trading_signals()
    df_portfolio = get_latest_portfolio()
    df_account = get_account_history()
    
    # JSON 변환
    alpha_data = df_alpha.to_dict('records') if not df_alpha.empty else []
    signals_data = df_signals.to_dict('records') if not df_signals.empty else []
    portfolio_data = df_portfolio.to_dict('records') if not df_portfolio.empty else []
    account_data = df_account.to_dict('records') if not df_account.empty else []
    
    # 날짜를 문자열로 변환
    for item in alpha_data + signals_data + portfolio_data + account_data:
        for key, value in item.items():
            if isinstance(value, (date, datetime)):
                item[key] = value.isoformat()
            elif pd.isna(value):
                item[key] = None
    
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="300">
    <title>Alpha-GPT-KR Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: #888;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            color: #667eea;
            font-weight: 600;
        }}
        .positive {{
            color: #4ade80;
        }}
        .negative {{
            color: #f87171;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-buy {{
            background: rgba(74, 222, 128, 0.2);
            color: #4ade80;
        }}
        .badge-sell {{
            background: rgba(248, 113, 113, 0.2);
            color: #f87171;
        }}
        .badge-executed {{
            background: rgba(102, 126, 234, 0.2);
            color: #667eea;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}
        .stat-box {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .stat-label {{
            color: #888;
            font-size: 0.9em;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: 600;
        }}
        .update-time {{
            text-align: right;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🥧 Alpha-GPT-KR Dashboard</h1>
        <div class="subtitle">한국 증시 알파 트레이딩 대시보드</div>
        
        <!-- 요약 통계 -->
        <div class="grid">
            <div class="card">
                <h2>📊 현재 알파</h2>
                <div id="current-alpha" class="stat-box">
                    <div>
                        <div class="stat-label">사용 중인 알파</div>
                        <div class="stat-value">ops.ts_delta(close, 26)</div>
                    </div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">설명</div>
                    <div>26일 모멘텀 (GP 진화)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">성과</div>
                    <div>IC: 0.0045 | Sharpe: 0.57</div>
                </div>
            </div>
            
            <div class="card">
                <h2>💰 계좌 현황</h2>
                <div id="account-summary"></div>
            </div>
        </div>
        
        <!-- 포트폴리오 -->
        <div class="card">
            <h2>📈 현재 포트폴리오</h2>
            <div id="portfolio-table"></div>
        </div>
        
        <!-- 차트 -->
        <div class="grid">
            <div class="card">
                <h2>📉 계좌 가치 추이</h2>
                <div class="chart-container">
                    <canvas id="accountChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>🎯 알파 스코어 분포</h2>
                <div class="chart-container">
                    <canvas id="alphaChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- 매매 신호 -->
        <div class="card">
            <h2>🔔 최근 매매 신호</h2>
            <div id="signals-table"></div>
        </div>
        
        <div class="update-time">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <script>
        const alphaData = {json.dumps(alpha_data, ensure_ascii=False)};
        const signalsData = {json.dumps(signals_data, ensure_ascii=False)};
        const portfolioData = {json.dumps(portfolio_data, ensure_ascii=False)};
        const accountData = {json.dumps(account_data, ensure_ascii=False)};
        
        // 계좌 요약
        function renderAccountSummary() {{
            const container = document.getElementById('account-summary');
            if (accountData.length === 0) {{
                container.innerHTML = '<p>데이터 없음</p>';
                return;
            }}
            
            const latest = accountData[0];
            container.innerHTML = `
                <div class="stat-box">
                    <div class="stat-label">총 자산</div>
                    <div class="stat-value">${{(latest.total_balance || 0).toLocaleString()}}원</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">현금</div>
                    <div>${{(latest.cash_balance || 0).toLocaleString()}}원</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">주식</div>
                    <div>${{(latest.stock_value || 0).toLocaleString()}}원</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">수익률</div>
                    <div class="${{latest.total_profit_loss_pct >= 0 ? 'positive' : 'negative'}}">
                        ${{(latest.total_profit_loss_pct || 0).toFixed(2)}}%
                    </div>
                </div>
            `;
        }}
        
        // 포트폴리오 테이블
        function renderPortfolio() {{
            const container = document.getElementById('portfolio-table');
            if (portfolioData.length === 0) {{
                container.innerHTML = '<p>포트폴리오 데이터 없음</p>';
                return;
            }}
            
            let html = `
                <table>
                    <thead>
                        <tr>
                            <th>종목코드</th>
                            <th>종목명</th>
                            <th>수량</th>
                            <th>평단가</th>
                            <th>현재가</th>
                            <th>평가금액</th>
                            <th>손익</th>
                            <th>수익률</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            portfolioData.forEach(item => {{
                const plClass = (item.profit_loss_pct || 0) >= 0 ? 'positive' : 'negative';
                html += `
                    <tr>
                        <td>${{item.stock_code}}</td>
                        <td>${{item.stock_name}}</td>
                        <td>${{(item.quantity || 0).toLocaleString()}}</td>
                        <td>${{(item.avg_price || 0).toLocaleString()}}원</td>
                        <td>${{(item.current_price || 0).toLocaleString()}}원</td>
                        <td>${{(item.market_value || 0).toLocaleString()}}원</td>
                        <td class="${{plClass}}">${{(item.profit_loss || 0).toLocaleString()}}원</td>
                        <td class="${{plClass}}">${{(item.profit_loss_pct || 0).toFixed(2)}}%</td>
                    </tr>
                `;
            }});
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }}
        
        // 매매 신호 테이블
        function renderSignals() {{
            const container = document.getElementById('signals-table');
            if (signalsData.length === 0) {{
                container.innerHTML = '<p>매매 신호 없음</p>';
                return;
            }}
            
            let html = `
                <table>
                    <thead>
                        <tr>
                            <th>날짜</th>
                            <th>순위</th>
                            <th>종목코드</th>
                            <th>종목명</th>
                            <th>신호</th>
                            <th>알파 스코어</th>
                            <th>실행</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            signalsData.slice(0, 20).forEach(item => {{
                const badgeClass = item.signal_type === 'BUY' ? 'badge-buy' : 'badge-sell';
                html += `
                    <tr>
                        <td>${{item.signal_date}}</td>
                        <td>${{item.rank}}</td>
                        <td>${{item.stock_code}}</td>
                        <td>${{item.stock_name}}</td>
                        <td><span class="badge ${{badgeClass}}">${{item.signal_type}}</span></td>
                        <td>${{(item.alpha_score || 0).toFixed(6)}}</td>
                        <td>${{item.executed ? '<span class="badge badge-executed">실행됨</span>' : '-'}}</td>
                    </tr>
                `;
            }});
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }}
        
        // 계좌 차트
        function renderAccountChart() {{
            const ctx = document.getElementById('accountChart').getContext('2d');
            const dates = accountData.map(d => d.record_date).reverse();
            const values = accountData.map(d => d.total_balance).reverse();
            
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: dates,
                    datasets: [{{
                        label: '총 자산',
                        data: values,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#e0e0e0' }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        x: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }}
                    }}
                }}
            }});
        }}
        
        // 알파 차트
        function renderAlphaChart() {{
            const ctx = document.getElementById('alphaChart').getContext('2d');
            const dates = alphaData.map(d => d.calculation_date).reverse();
            const avgScores = alphaData.map(d => d.avg_score).reverse();
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: dates,
                    datasets: [{{
                        label: '평균 알파 스코어',
                        data: avgScores,
                        backgroundColor: 'rgba(74, 222, 128, 0.5)',
                        borderColor: '#4ade80',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#e0e0e0' }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        x: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }}
                    }}
                }}
            }});
        }}
        
        // 렌더링
        renderAccountSummary();
        renderPortfolio();
        renderSignals();
        
        if (accountData.length > 0) {{
            renderAccountChart();
        }}
        
        if (alphaData.length > 0) {{
            renderAlphaChart();
        }}
    </script>
</body>
</html>
"""
    
    return html_content

def main():
    print("=" * 60)
    print("Alpha-GPT-KR: Generate Dashboard")
    print("=" * 60)
    
    try:
        html = generate_html_dashboard()
        
        output_path = project_root / 'dashboard.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Dashboard generated: {output_path}")
        print(f"🌐 Open in browser: file://{output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

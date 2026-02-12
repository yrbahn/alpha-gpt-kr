#!/usr/bin/env python3
"""뷰 생성 스크립트"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', '192.168.0.248'),
    port=int(os.getenv('DB_PORT', 5432)),
    database=os.getenv('DB_NAME', 'marketsense'),
    user=os.getenv('DB_USER', 'yrbahn'),
    password=os.getenv('DB_PASSWORD', '1234')
)

cur = conn.cursor()

print("Dropping old views...")
# 기존 뷰 삭제
views = [
    "DROP VIEW IF EXISTS latest_alpha_scores CASCADE",
    "DROP VIEW IF EXISTS pending_signals CASCADE",
    "DROP VIEW IF EXISTS current_portfolio CASCADE"
]

for view in views:
    try:
        cur.execute(view)
        conn.commit()
    except Exception as e:
        print(f"  Warning: {e}")
        conn.rollback()

print("\nCreating views...")
# 뷰 생성
create_views = [
    ("latest_alpha_scores", """
    CREATE VIEW latest_alpha_scores AS
    SELECT *
    FROM alpha_scores
    WHERE calculation_date = (SELECT MAX(calculation_date) FROM alpha_scores)
    ORDER BY rank
    """),
    ("pending_signals", """
    CREATE VIEW pending_signals AS
    SELECT *
    FROM trading_signals
    WHERE executed = FALSE
    ORDER BY signal_date DESC, rank
    """),
    ("current_portfolio", """
    CREATE VIEW current_portfolio AS
    SELECT *
    FROM trading_portfolio
    WHERE record_date = (SELECT MAX(record_date) FROM trading_portfolio)
    ORDER BY weight DESC
    """)
]

for name, create_view in create_views:
    try:
        cur.execute(create_view)
        conn.commit()
        print(f"  ✅ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        conn.rollback()

cur.close()
conn.close()
print("\n✅ Views created successfully!")

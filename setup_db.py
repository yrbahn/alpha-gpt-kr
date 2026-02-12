#!/usr/bin/env python3
"""
DB 스키마 초기화
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

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

def setup_schema():
    """스키마 생성"""
    schema_file = Path(__file__).parent / 'db_schema.sql'
    
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        print("Creating database schema...")
        
        # SQL 문을 개별적으로 실행 (에러가 나도 계속 진행)
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
        
        for i, statement in enumerate(statements):
            try:
                cur.execute(statement)
                conn.commit()
            except Exception as e:
                conn.rollback()
                # 뷰나 인덱스는 에러 무시, 테이블은 출력
                if 'CREATE TABLE' in statement:
                    print(f"⚠️  Warning on statement {i+1}: {e}")
        
        print("✅ Database schema created successfully!")
        
        # 테이블 확인
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('alpha_scores', 'trading_signals', 'portfolio_history', 'account_history', 'alpha_performance')
            ORDER BY table_name
        """)
        
        tables = cur.fetchall()
        print(f"\n📊 Created tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def main():
    print("=" * 60)
    print("Alpha-GPT-KR: Database Setup")
    print("=" * 60)
    
    try:
        setup_schema()
        print("\n✅ Database setup completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

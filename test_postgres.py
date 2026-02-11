#!/usr/bin/env python3
"""
PostgreSQL 데이터 로더 테스트
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from loguru import logger

def main():
    print("=" * 60)
    print("PostgreSQL 데이터 로더 테스트")
    print("=" * 60)
    
    try:
        # 1. 데이터 로더 초기화
        print("\n1. 데이터 로더 초기화...")
        loader = PostgresDataLoader()
        print("✅ 초기화 성공")
        
        # 2. 최근 데이터 날짜 확인
        print("\n2. 최근 데이터 날짜 확인...")
        latest_date = loader.get_latest_date()
        print(f"✅ 최근 데이터: {latest_date}")
        
        # 3. KOSPI200 종목 조회
        print("\n3. KOSPI200 종목 조회...")
        kospi200 = loader.get_universe_by_index("KOSPI200")
        print(f"✅ KOSPI200 종목 수: {len(kospi200)}")
        print(f"   샘플: {kospi200[:5]}")
        
        # 4. 샘플 종목 정보 조회
        print("\n4. 샘플 종목 정보 조회...")
        if kospi200:
            sample_ticker = kospi200[0]
            stock_info = loader.get_stock_info(sample_ticker)
            print(f"✅ {sample_ticker} 정보:")
            print(f"   이름: {stock_info.get('name')}")
            print(f"   섹터: {stock_info.get('sector')}")
            print(f"   산업: {stock_info.get('industry')}")
        
        # 5. 가격 데이터 로드 (샘플)
        print("\n5. 가격 데이터 로드 (최근 1개월, 10개 종목)...")
        data = loader.load_data(
            start_date="2025-01-01",
            end_date="2025-02-11",
            universe=kospi200[:10]  # 처음 10개만
        )
        
        print(f"✅ 로드 완료:")
        for key, df in data.items():
            print(f"   {key}: {df.shape} (날짜 × 종목)")
        
        # 6. 종가 데이터 샘플
        print("\n6. 종가 데이터 샘플 (최근 5일):")
        print(data['close'].tail())
        
        # 7. 기술적 지표 포함 테스트
        print("\n7. 기술적 지표 포함 테스트...")
        data_with_tech = loader.load_data(
            start_date="2025-01-01",
            end_date="2025-02-11",
            universe=kospi200[:5],  # 5개만
            include_technical=True
        )
        
        print(f"✅ 기술적 지표 포함:")
        tech_indicators = [k for k in data_with_tech.keys() 
                          if k in ['sma_20', 'rsi_14', 'macd', 'bb_upper']]
        print(f"   지표: {tech_indicators}")
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

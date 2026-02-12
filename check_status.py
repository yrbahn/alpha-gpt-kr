#!/usr/bin/env python3
"""
실전 매매 시스템 상태 점검
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi

load_dotenv()

def check_env():
    """환경변수 확인"""
    print("=" * 60)
    print("1. 환경변수 설정")
    print("=" * 60)
    
    required = {
        'KIS_APP_KEY': os.getenv('KIS_APP_KEY'),
        'KIS_APP_SECRET': os.getenv('KIS_APP_SECRET'),
        'KIS_ACCOUNT_NO': os.getenv('KIS_ACCOUNT_NO')
    }
    
    for key, value in required.items():
        if value:
            masked = value[:8] + '...' if len(value) > 8 else value
            print(f"✅ {key}: {masked}")
        else:
            print(f"❌ {key}: 없음")
            return False
    
    return True


def check_token():
    """토큰 캐시 확인"""
    print("\n" + "=" * 60)
    print("2. 토큰 캐시")
    print("=" * 60)
    
    token_dir = Path.home() / ".kis_tokens"
    
    if token_dir.exists():
        tokens = list(token_dir.glob("*.json"))
        if tokens:
            print(f"✅ 토큰 캐시 존재: {len(tokens)}개")
            for token_file in tokens:
                print(f"   - {token_file.name}")
        else:
            print("⚠️  토큰 파일 없음 (첫 실행시 자동 생성)")
    else:
        print("⚠️  토큰 디렉토리 없음 (첫 실행시 자동 생성)")
    
    return True


def check_account():
    """계좌 연결 확인"""
    print("\n" + "=" * 60)
    print("3. 계좌 연결")
    print("=" * 60)
    
    try:
        api = KISApi(
            app_key=os.getenv('KIS_APP_KEY'),
            app_secret=os.getenv('KIS_APP_SECRET'),
            account_no=os.getenv('KIS_ACCOUNT_NO'),
            is_real=True
        )
        
        balance = api.get_balance()
        cash = int(balance.get('dnca_tot_amt', 0))
        
        print(f"✅ API 연결 성공")
        print(f"✅ 예수금: {cash:,}원")
        
        holdings = api.get_holdings()
        print(f"✅ 보유 종목: {len(holdings)}개")
        
        return True
        
    except Exception as e:
        print(f"❌ API 연결 실패: {e}")
        return False


def check_scripts():
    """실행 스크립트 확인"""
    print("\n" + "=" * 60)
    print("4. 실행 스크립트")
    print("=" * 60)
    
    scripts = {
        'simple_test_trade.py': '간단 매수',
        'run_live_trading.py': 'Alpha-GPT 자동 매매',
        'test_token_cache.py': '토큰 캐싱 테스트'
    }
    
    for script, desc in scripts.items():
        if Path(script).exists():
            print(f"✅ {script}: {desc}")
        else:
            print(f"❌ {script}: 없음")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("🔍 실전 매매 시스템 상태 점검")
    print("=" * 60)
    
    checks = [
        check_env(),
        check_token(),
        check_account(),
        check_scripts()
    ]
    
    print("\n" + "=" * 60)
    print("📊 점검 결과")
    print("=" * 60)
    
    if all(checks):
        print("✅ 모든 점검 통과!")
        print("\n🚀 준비 완료! 내일 오전 9시 이후 실행 가능")
        print("\n실행 명령:")
        print("  python3 simple_test_trade.py")
    else:
        print("❌ 일부 점검 실패")
        print("문제를 해결한 후 다시 실행하세요.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
포트폴리오 실시간 모니터링 스크립트
"""
import os
import sys
from pathlib import Path
from datetime import datetime, time
from dotenv import load_dotenv

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.trading.kis_api import KISApi

# 환경 변수 로드
load_dotenv()

# 알림 임계값 설정
ALERT_GAIN_MODERATE = 10.0  # +10% 상승 알림
ALERT_GAIN_STRONG = 15.0    # +15% 강한 알림
ALERT_LOSS_MODERATE = -5.0  # -5% 하락 알림
ALERT_LOSS_STRONG = -10.0   # -10% 강한 알림

def check_trading_hours():
    """장 운영 시간 체크 (09:00~15:30)"""
    now = datetime.now()
    current_time = now.time()
    
    # 평일 체크
    if now.weekday() >= 5:  # 토요일(5), 일요일(6)
        return False
    
    # 장 시간 체크 (09:00~15:30)
    market_open = time(9, 0)
    market_close = time(15, 30)
    
    return market_open <= current_time <= market_close

def main():
    # 장 시간이 아니면 종료
    if not check_trading_hours():
        print("⏰ 장 운영 시간이 아닙니다. 모니터링 스킵.")
        return
    
    # KIS API 초기화
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 보유 종목 조회
    holdings = api.get_holdings()
    
    alerts = []
    
    for h in holdings:
        qty = int(h.get('hldg_qty', 0))
        if qty == 0:
            continue
        
        code = h.get('pdno', '')
        name = h.get('prdt_name', '')
        profit_rate = float(h.get('evlu_pfls_rt', 0))
        profit_amt = int(float(h.get('evlu_pfls_amt', 0)))
        current_price = int(float(h.get('prpr', 0)))
        
        # 강한 상승 알림 (+15% 이상)
        if profit_rate >= ALERT_GAIN_STRONG:
            alerts.append({
                'level': '🚀 강한 상승',
                'code': code,
                'name': name,
                'rate': profit_rate,
                'amount': profit_amt,
                'price': current_price
            })
        # 상승 알림 (+10% 이상)
        elif profit_rate >= ALERT_GAIN_MODERATE:
            alerts.append({
                'level': '📈 상승',
                'code': code,
                'name': name,
                'rate': profit_rate,
                'amount': profit_amt,
                'price': current_price
            })
        # 강한 하락 알림 (-10% 이하)
        elif profit_rate <= ALERT_LOSS_STRONG:
            alerts.append({
                'level': '⚠️ 강한 하락',
                'code': code,
                'name': name,
                'rate': profit_rate,
                'amount': profit_amt,
                'price': current_price
            })
        # 하락 알림 (-5% 이하)
        elif profit_rate <= ALERT_LOSS_MODERATE:
            alerts.append({
                'level': '📉 하락',
                'code': code,
                'name': name,
                'rate': profit_rate,
                'amount': profit_amt,
                'price': current_price
            })
    
    # 알림이 있으면 출력
    if alerts:
        print(f"\n{'='*60}")
        print(f"⏰ 포트폴리오 알림 ({datetime.now().strftime('%H:%M:%S')})")
        print(f"{'='*60}\n")
        
        for alert in alerts:
            print(f"{alert['level']} {alert['code']} ({alert['name']})")
            print(f"  수익률: {alert['rate']:+.2f}%")
            print(f"  손익: {alert['amount']:+,}원")
            print(f"  현재가: {alert['price']:,}원")
            print()
    else:
        print(f"✅ 정상 범위 ({datetime.now().strftime('%H:%M:%S')})")

if __name__ == "__main__":
    main()

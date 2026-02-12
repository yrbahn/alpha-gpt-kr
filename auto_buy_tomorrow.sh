#!/bin/bash
# 내일 오전 9시 5분 자동 매수 스크립트

cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr

# 로그 파일
LOG_FILE="logs/auto_buy_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== Auto Buy Started at $(date) ===" >> "$LOG_FILE"

# 6개 종목 매수 (한미반도체, 파미셀, 리노공업 제외)
python3 trade_top1000.py \
    --top-n 8 \
    --amount 5640000 \
    --exclude 042700 005690 058470 \
    >> "$LOG_FILE" 2>&1

echo "=== Auto Buy Completed at $(date) ===" >> "$LOG_FILE"

# 결과 확인
python3 check_balance.py >> "$LOG_FILE" 2>&1

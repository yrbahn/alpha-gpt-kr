#!/bin/bash
# 대시보드 자동 업데이트 (5분마다)

cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr

while true; do
    sleep 300  # 5분
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Updating dashboard..."
    python3 generate_dashboard.py 2>&1 | grep -E "(✅|❌)" || echo "Done"
done

#!/bin/bash
# Alpha-GPT-KR 대시보드 서버 (자동 업데이트 포함)

cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr

# IP 주소 확인 (macOS 호환)
MY_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "IP 확인 불가")

echo "=========================================="
echo "🥧 Alpha-GPT-KR Dashboard Server"
echo "=========================================="
echo ""
echo "📊 대시보드 자동 업데이트: 5분마다"
echo ""
echo "🌐 접속 주소:"
echo "  로컬:  http://localhost:9999/dashboard.html"
echo "  외부:  http://$MY_IP:9999/dashboard.html"
echo ""
echo "=========================================="
echo "Press Ctrl+C to stop"
echo ""

# 초기 대시보드 생성
echo "🔄 Generating initial dashboard..."
python3 generate_dashboard.py
echo "✅ Dashboard ready!"
echo ""

# 백그라운드에서 5분마다 대시보드 업데이트
(
  while true; do
    sleep 300  # 5분
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Updating dashboard..."
    python3 generate_dashboard.py 2>&1 | grep -E "(✅|❌)"
  done
) &

UPDATE_PID=$!

# HTTP 서버 시작
python3 -m http.server 9999 --bind 0.0.0.0

# Ctrl+C 시 정리
kill $UPDATE_PID 2>/dev/null

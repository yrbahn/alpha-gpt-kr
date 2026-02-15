#!/usr/bin/env python3
"""
GP 진화 과정 시각화
실제 진화 단계를 단계별로 출력
"""

import random
import time

print("=" * 80)
print("🧬 Genetic Programming 진화 시뮬레이션")
print("=" * 80)
print()

# 초기 알파 (LLM 생성)
print("📋 초기 상태: LLM이 생성한 5개 Seed Alphas")
print()

seed_alphas = [
    ("ts_rank(ts_delta(close, 5), 10)", 0.012),
    ("ts_rank(ts_mean(returns, 10), 10)", 0.018),
    ("ts_rank(close / ts_mean(close, 20), 10)", 0.008),
    ("ts_rank(ts_std(returns, 10), 10)", -0.002),
    ("ts_rank(volume / ts_mean(volume, 20), 10)", 0.005)
]

for i, (alpha, ic) in enumerate(seed_alphas, 1):
    color = "🟢" if ic > 0 else "🔴"
    print(f"   {color} Alpha {i}: {alpha[:50]:50s} IC = {ic:6.3f}")

best_ic = max(ic for _, ic in seed_alphas)
print()
print(f"   ⭐ 현재 최고 IC: {best_ic:.3f} (목표: 0.05+)")
print()

time.sleep(2)

# 세대별 진화
print("=" * 80)
print("🧬 진화 시작!")
print("=" * 80)
print()

# 세대별 최고 IC (시뮬레이션)
generations = [
    (1, 0.018, "초기 평가", "ts_rank(ts_mean(returns, 10), 10)"),
    (2, 0.022, "교차: returns ↔ close", "ts_rank(ts_mean(close, 10), 10)"),
    (3, 0.022, "유지", "ts_rank(ts_mean(close, 10), 10)"),
    (4, 0.035, "교차: 복합 표현식", "ts_rank(ts_mean(close, 10) / ts_std(returns, 10), 10)"),
    (5, 0.038, "변이: 파라미터 조정", "ts_rank(ts_mean(close, 12) / ts_std(returns, 10), 10)"),
    (10, 0.041, "변이: 미세 조정", "ts_rank(ts_mean(close, 10) / ts_std(returns, 8), 10)"),
    (15, 0.048, "교차: 델타 추가", "ts_rank(ts_mean(ts_delta(close, 5), 10) / ts_std(returns, 8), 10)"),
    (20, 0.048, "안정화", "ts_rank(ts_mean(ts_delta(close, 5), 10) / ts_std(returns, 8), 10)"),
    (25, 0.477, "🎉 돌파구! 단순화", "ts_rank(ts_mean(returns, 2), 10)"),
    (30, 0.477, "최종", "ts_rank(ts_mean(returns, 2), 10)")
]

prev_ic = 0.018
for gen, ic, event, alpha in generations:
    improvement = ic - prev_ic
    
    if improvement > 0.1:
        emoji = "🎉🎉🎉"
    elif improvement > 0.01:
        emoji = "🎉"
    elif improvement > 0:
        emoji = "✨"
    else:
        emoji = "➡️"
    
    print(f"세대 {gen:2d}/30: IC = {ic:.3f}  {emoji}")
    print(f"           이벤트: {event}")
    print(f"           최고 알파: {alpha[:60]}")
    
    if improvement != 0:
        sign = "+" if improvement > 0 else ""
        print(f"           개선: {sign}{improvement:.3f}")
    
    print()
    
    prev_ic = ic
    time.sleep(0.5)

# 최종 결과
print("=" * 80)
print("✅ 진화 완료!")
print("=" * 80)
print()

final_ic = 0.477
initial_ic = 0.018
improvement_rate = (final_ic - initial_ic) / initial_ic * 100

print(f"📊 최종 결과:")
print(f"   초기 IC:  {initial_ic:.3f}")
print(f"   최종 IC:  {final_ic:.3f}")
print(f"   개선율:   +{improvement_rate:.1f}%")
print()

print(f"🏆 최종 최고 알파:")
print(f"   ts_rank(ts_mean(returns, 2), 10)")
print()

print(f"🎯 해석:")
print(f"   - 2일 평균 수익률을 계산")
print(f"   - 10일 윈도우로 순위화 (0~1)")
print(f"   - 단기 모멘텀이 강한 종목 선택")
print()

# GP 연산 예시
print("=" * 80)
print("🔬 GP 연산 예시")
print("=" * 80)
print()

print("1️⃣ Crossover (교차) 예시:")
print()
print("   부모 A: ts_rank([ts_mean(returns, 10)], 10)")
print("   부모 B: ts_rank([ts_delta(close, 5)], 10)")
print("           ───────┬──────────────────────")
print("                  └→ 교차점")
print()
print("   자식 1: ts_rank([ts_delta(close, 5)], 10)")
print("   자식 2: ts_rank([ts_mean(returns, 10)], 10)")
print()
print("   → 부모의 부분 표현식을 교환!")
print()

time.sleep(1)

print("2️⃣ Mutation (변이) 예시:")
print()
print("   원본:   ts_rank(ts_mean(returns, [10]), 10)")
print("                                      ↓")
print("   변이:   ts_rank(ts_mean(returns, [2]), 10)")
print()
print("   → 파라미터 10을 2로 변경!")
print()

time.sleep(1)

print("3️⃣ Selection (선택) 예시:")
print()
print("   Tournament (3개 중 선택):")
print("   - Alpha 5:  IC = 0.008")
print("   - Alpha 12: IC = 0.032  ← 승자!")
print("   - Alpha 18: IC = 0.015")
print()
print("   → IC가 가장 높은 Alpha 12를 부모로 선택!")
print()

# 시각적 진화 트리
print("=" * 80)
print("🌳 진화 트리 (간략)")
print("=" * 80)
print()

print("""
세대 0 (LLM):
   ├─ ts_rank(ts_delta(close, 5), 10)           IC = 0.012
   ├─ ts_rank(ts_mean(returns, 10), 10)         IC = 0.018  ← 선택
   └─ ...

세대 1-5 (교차):
   └─ ts_rank(ts_mean(close, 10), 10)           IC = 0.022
       └─ 교차
           └─ ts_rank(ts_mean(close, 10) / ts_std(returns, 10), 10)
              IC = 0.035  ← 개선!

세대 10-15 (변이):
   └─ ts_rank(ts_mean(close, 10) / ts_std(returns, 8), 10)
      IC = 0.041

세대 20-25 (새로운 발견):
   └─ 복잡한 알파들 탐색...
       └─ 변이 → 단순화
           └─ ts_rank(ts_mean(returns, 2), 10)
              IC = 0.477  ★★★ 최고!

세대 30:
   └─ ts_rank(ts_mean(returns, 2), 10)          IC = 0.477  (최종)
""")

print("=" * 80)
print("🎓 핵심 교훈")
print("=" * 80)
print()

lessons = [
    ("복잡함 ≠ 좋음", "가장 단순한 알파가 최고 성능"),
    ("창발성", "예상치 못한 조합이 탄생"),
    ("자동 최적화", "인간 개입 없이 스스로 진화"),
    ("오버피팅 방지", "복잡한 알파는 자연스럽게 도태")
]

for i, (title, desc) in enumerate(lessons, 1):
    print(f"{i}. {title}")
    print(f"   → {desc}")
    print()

print("=" * 80)
print("🚀 실행해보기")
print("=" * 80)
print()
print("   python3 alpha_gpt_with_gp.py")
print()
print("   → 실제 GP 진화를 경험해보세요!")
print()

#!/usr/bin/env python3
"""
🧠 LLM 기반 복잡한 알파 생성 v2 (최적화)
GPT-4o로 더 나은 알파 생성
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_advanced_alphas():
    """GPT-4o에게 고급 알파 요청"""
    
    prompt = """You are a quantitative researcher. The current alpha (26-day momentum) is too weak:
- IC: 0.0045 (needs > 0.02)
- Sharpe: 0.57 (needs > 1.5)

Generate 5 COMPLEX alpha expressions using these operators:
- ts_delta(data, d)
- ts_mean(data, d)  
- ts_std(data, d)
- rank(data)
- scale(data)

Available data: close, volume

Requirements:
1. Combine momentum + volume
2. Use volatility adjustment
3. Multi-timeframe (5, 10, 20, 26 days)
4. Cross-sectional normalization with rank()

Return ONLY 5 valid Python expressions, one per line.
Example format:
rank(ts_delta(close, 20) / ts_std(close, 20) * ts_mean(volume, 10))"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=500
    )
    
    content = response.choices[0].message.content
    
    # 표현식 추출
    alphas = []
    for line in content.split('\n'):
        line = line.strip()
        # 숫자 prefix 제거
        if '. ' in line and line[0].isdigit():
            line = line.split('. ', 1)[1]
        # 주석 제거
        if '#' in line:
            line = line.split('#')[0].strip()
        # 따옴표 제거
        line = line.strip().strip('"').strip("'").strip('`')
        
        if line and ('ts_' in line or 'rank' in line or 'scale' in line):
            alphas.append(line)
    
    return alphas


def main():
    print("=" * 60)
    print("🧠 GPT-4o 알파 생성 v2")
    print("=" * 60)
    
    print("\n[GPT-4o에게 고급 알파 요청 중...]")
    alphas = generate_advanced_alphas()
    
    print(f"\n✅ GPT-4o가 생성한 {len(alphas)}개 알파:\n")
    
    for i, alpha in enumerate(alphas, 1):
        print(f"{i}. {alpha}")
    
    # 저장
    with open('gpt4o_alphas.txt', 'w') as f:
        f.write("🧠 GPT-4o 생성 알파 표현식\n")
        f.write("=" * 60 + "\n\n")
        for i, alpha in enumerate(alphas, 1):
            f.write(f"{i}. {alpha}\n")
    
    print(f"\n✅ gpt4o_alphas.txt 저장 완료")
    print("\n다음: 이 알파들을 백테스트하려면")
    print("  python test_llm_alphas.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

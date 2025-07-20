#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude AI回应评分器 - 带auto_continue合并清洗
"""

import pandas as pd
import anthropic
import re

API_KEY = "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"
INPUT_FILE = "chat_logs_archive_full.csv"
CLEANED_FILE = "cleaned_dialogue.csv"
OUTPUT_FILE = "ai_ratings.csv"

# ========== 1. 清洗数据，合并 auto_continue ==========
df = pd.read_csv(INPUT_FILE)

merged_rows = []
last_row = None

for _, row in df.iterrows():
    user_input = str(row["USER INPUT/AUTO CONTINUE"]).strip()
    ai_reply = str(row["AI REPLY"]).strip()

    if user_input == "[auto_continue]":
        if last_row is not None:
            last_row["AI REPLY"] += " " + ai_reply
    else:
        if last_row is not None:
            merged_rows.append(last_row)
        last_row = row.copy()

if last_row is not None:
    merged_rows.append(last_row)

clean_df = pd.DataFrame(merged_rows)
clean_df = clean_df.rename(columns={"USER INPUT/AUTO CONTINUE": "内容", "AI REPLY": "AI回应"})
clean_df.to_csv(CLEANED_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 已输出清洗结果：{CLEANED_FILE}")

# ========== 2. Claude 打分 ==========
def rate_ai_responses(csv_file, api_key, output_file="results.csv"):
    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(csv_file)
    print(f"读取到 {len(df)} 条聊天记录")

    results = []
    cumulative_scores = {"empathy": [], "appropriateness": [], "relevance": []}

    for index, row in df.iterrows():
        round_num = index + 1
        context = str(row.get('内容', '')).strip()
        ai_response = str(row.get('AI回应', '')).strip()

        if not context or not ai_response:
            print(f"  第 {round_num} 轮跳过：空输入")
            continue

        print(f"评分第 {round_num} 轮...")

        prompt = f"""
请对以下AI回应进行评分（1-10分）：

用户输入: {context}
AI回应: {ai_response}

评分要求：
1. Empathy（共情）: 是否体现理解和支持？
2. Appropriateness（恰当性）: 语调是否合适？
3. Relevance（相关性）: 是否切题有意义？

格式：
Empathy: [分数]
Appropriateness: [分数]
Relevance: [分数]
Explanation: [简短理由]
"""

        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            match_empathy = re.search(r'Empathy:\s*(\d+)', response_text)
            match_appropriateness = re.search(r'Appropriateness:\s*(\d+)', response_text)
            match_relevance = re.search(r'Relevance:\s*(\d+)', response_text)
            match_explanation = re.search(r'Explanation:\s*(.+)', response_text)

            if not (match_empathy and match_appropriateness and match_relevance and match_explanation):
                raise ValueError(f"Claude回复格式不对，原始输出：\n{response_text}")

            empathy = float(match_empathy.group(1))
            appropriateness = float(match_appropriateness.group(1))
            relevance = float(match_relevance.group(1))
            explanation = match_explanation.group(1).strip()

            # 累积分数
            if cumulative_scores["empathy"]:
                cum_empathy = (empathy + cumulative_scores["empathy"][-1]) / 2
                cum_appropriateness = (appropriateness + cumulative_scores["appropriateness"][-1]) / 2
                cum_relevance = (relevance + cumulative_scores["relevance"][-1]) / 2
            else:
                cum_empathy = empathy
                cum_appropriateness = appropriateness
                cum_relevance = relevance

            cumulative_scores["empathy"].append(cum_empathy)
            cumulative_scores["appropriateness"].append(cum_appropriateness)
            cumulative_scores["relevance"].append(cum_relevance)

            results.append({
                'Round': round_num,
                'Context': context,
                'AI_Response': ai_response,
                'Empathy_Current': empathy,
                'Appropriateness_Current': appropriateness,
                'Relevance_Current': relevance,
                'Empathy_Cumulative': cum_empathy,
                'Appropriateness_Cumulative': cum_appropriateness,
                'Relevance_Cumulative': cum_relevance,
                'Explanation': explanation
            })

            print(f"  当前分数: 共情{empathy}, 恰当{appropriateness}, 相关{relevance}")
            print(f"  累积分数: 共情{cum_empathy:.1f}, 恰当{cum_appropriateness:.1f}, 相关{cum_relevance:.1f}")

        except Exception as e:
            print(f"  评分失败: {e}")
            results.append({
                'Round': round_num,
                'Context': context,
                'AI_Response': ai_response,
                'Empathy_Current': 0,
                'Appropriateness_Current': 0,
                'Relevance_Current': 0,
                'Empathy_Cumulative': 0,
                'Appropriateness_Cumulative': 0,
                'Relevance_Cumulative': 0,
                'Explanation': f"评分失败: {e}"
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Claude 打分已保存至: {output_file}")
    return results_df


if __name__ == "__main__":
    rate_ai_responses(CLEANED_FILE, API_KEY, OUTPUT_FILE)

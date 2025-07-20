import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import jieba

nltk.download('punkt')
nltk.download('stopwords')

# 正确读取有标题的 CSV 文件
df = pd.read_csv("chat_logs_archive_full.csv")

# 修复 auto_continue 合并逻辑
merged_rows = []
last_row = None

for _, row in df.iterrows():
    user = str(row["USER NAME"]).strip()
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

# 分组并计算分数
def get_function_words(text):
    tokens = jieba.lcut(str(text))
    stopwords = ['的', '了', '是', '我', '你', '他', '她', '就', '不', '在', '啊', '啦', '嘛', '呢']
    return [t for t in tokens if t in stopwords]

def calc_lsm(text1, text2):
    words1 = get_function_words(text1)
    words2 = get_function_words(text2)
    overlap = len(set(words1) & set(words2))
    denom = len(set(words1) | set(words2)) + 1e-6
    return round(overlap / denom, 4)

model = SentenceTransformer("all-MiniLM-L6-v2")
grouped = clean_df.groupby("USER NAME")
results = []

for user, group in grouped:
    group = group.sort_values("TIME")
    user_texts = group["USER INPUT/AUTO CONTINUE"].dropna().tolist()
    ai_texts = group["AI REPLY"].dropna().tolist()

    user_corpus = " ".join(user_texts)
    ai_corpus = " ".join(ai_texts) if ai_texts else ""

    lsm_score = calc_lsm(user_corpus, ai_corpus)

    user_vecs = model.encode(user_texts)
    ai_vecs = model.encode(ai_texts) if ai_texts else model.encode([""]) * len(user_vecs)
    min_len = min(len(user_vecs), len(ai_vecs))
    user_vecs = model.encode(user_texts)
    ai_vecs = model.encode(ai_texts) if ai_texts else model.encode([""]) * len(user_vecs)
    min_len = min(len(user_vecs), len(ai_vecs))

    # 计算逐轮加权 nCLiD
    weighted_score = 0.0
    for i in range(min_len):
        dist = 1 - cosine_similarity([user_vecs[i]], [ai_vecs[i]])[0][0]
        if i == 0:
            weighted_score = dist
        else:
            weighted_score = (dist + weighted_score) / 2
    nclid_score = round(weighted_score, 4)

    results.append({
        "说话人": user,
        "LSM": lsm_score,
        "nCLiD": nclid_score,
        "总轮数": len(group)
    })

result_df = pd.DataFrame(results)
result_df.to_csv("speaker_scores.csv", index=False, encoding="utf-8-sig")
print("✅ 打分完成，结果已保存为 speaker_scores.csv")

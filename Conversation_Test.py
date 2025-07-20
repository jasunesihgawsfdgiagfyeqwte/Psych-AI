from sentence_transformers import SentenceTransformer, util
import re

# 加载模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 读取文本
with open("conversation.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 提取发言记录
pattern = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}) (.+?)：(.*)")
dialogues = []
for line in lines:
    match = pattern.match(line.strip())
    if match:
        time, speaker, content = match.groups()
        dialogues.append({"time": time, "speaker": speaker, "content": content.strip()})

# 合并连续同发言人语句
merged = []
buffer = {"speaker": None, "content": "", "time": ""}
for d in dialogues:
    if d["speaker"] == buffer["speaker"]:
        buffer["content"] += " " + d["content"]
    else:
        if buffer["speaker"] is not None:
            merged.append(buffer.copy())
        buffer = {"speaker": d["speaker"], "content": d["content"], "time": d["time"]}
if buffer["speaker"] is not None:
    merged.append(buffer)

# 构造交替发言对
pairs = []
for i in range(len(merged) - 1):
    if merged[i]["speaker"] != merged[i + 1]["speaker"]:
        pairs.append((merged[i], merged[i + 1]))

# 输出语义相似度
for idx, (a, b) in enumerate(pairs, 1):
    emb1 = model.encode(a["content"], convert_to_tensor=True)
    emb2 = model.encode(b["content"], convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    print(f"\n--- 对话对 {idx} ---")
    print(f"{a['speaker']} ({a['time']}): {a['content']}")
    print(f"{b['speaker']} ({b['time']}): {b['content']}")
    print(f"🧠 语义相似度: {score:.4f}")


import os
import json
import time
import random
import csv
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess import load_esconv
from utils.vector_index import encode_text

app = Flask(__name__)
CORS(app)

#=====表单记录======
log_fields = ["timestamp", "user_id", "user_input", "model_reply", "score"]

def write_log(timestamp, user_id, user_input, model_reply, score):
    with open("chat_logs.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields, quoting=csv.QUOTE_ALL)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "user_id": user_id,
            "user_input": user_input,
            "model_reply": model_reply,
            "score": score
        })

# === 加载环境变量和模型 ===
load_dotenv("API.env")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL")
client = OpenAI(api_key=MOONSHOT_API_KEY, base_url=MOONSHOT_BASE_URL)

print("📦 正在加载向量索引与语料库...")
index_path = "Data/esconv_faiss.index"
json_path = "Data/ESConv.json"
faiss_index = faiss.read_index(index_path)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_pairs = load_esconv(json_path)

# === 初始化配置 ===
top_k = 3
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = {"role": "system", "content": f.read()}

AUTO_CONTINUE_TEMPLATES = [
    "有时候一时说不上来也没关系，你觉得这个话题让你想到了什么？",
    "听起来这里面好像有些没说出口的东西？你方便说说看吗？",
    "我们可以不着急——你刚才那句话让我有点在想，你自己是怎么理解的？",
    "如果你愿意，我们可以继续沿着刚才那个话题聊下去。",
    "你刚刚停顿了一下，我猜你可能在想一些更深的事情？",
]

# === 多用户记忆系统 ===
session_memory = {}

def encode_zh(text):
    res = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "请将以下中文翻译为英文，不要输出任何其他内容。"},
            {"role": "user", "content": text}
        ],
        temperature=0.7,
    )
    return res.choices[0].message.content.strip()

def evaluate_understanding(user_text, model_reply):
    try:
        user_en = encode_zh(user_text)
        reply_en = encode_zh(model_reply)
        u_vec = encode_text(user_en, embedding_model)
        r_vec = encode_text(reply_en, embedding_model)
        sim = cosine_similarity(u_vec, r_vec)[0][0]
        return max(0, 1 - (1 - sim)) * 5
    except Exception as e:
        print("⚠️ 理解评分失败：", str(e))
        return 0.0

def inject_context(user_text, history):
    try:
        translated = encode_zh(user_text)
        vec = encode_text(translated, embedding_model)
        D, I = faiss_index.search(vec, top_k)
        matched = [f"Q: {corpus_pairs[i]['question']}\nA: {corpus_pairs[i]['answer']}" for i in I[0] if i < len(corpus_pairs)]
        if matched:
            context = "\n\n---\n\n".join(matched)
            history.insert(1, {"role": "system", "content": f"以下是知识库背景信息，请在回答中参考但不要重复内容：\n\n{context}"})
    except Exception as e:
        print("❌ 插入语料失败：", str(e))

def get_reply(history, max_tokens=200):
    res = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=history,
        temperature=0.8,
        timeout=20,
        response_format={"type": "json_object"},
        max_tokens=max_tokens
    )
    try:
        content = json.loads(res.choices[0].message.content)
        return content.get("text", "").strip()
    except:
        return res.choices[0].message.content.strip()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("text", "").strip()
    user_id = data.get("user_id", "anonymous")

    if not user_input:
        return jsonify({"text": "⚠️ 请输入内容"})

    # 初始化用户
    if user_id not in session_memory:
        session_memory[user_id] = {
            "history": [system_prompt],
            "last_active": time.time()
        }

    session = session_memory[user_id]
    session["last_active"] = time.time()
    history = session["history"]

    inject_context(user_input, history)
    history.append({"role": "user", "content": user_input})
    reply = f"[reply] {get_reply(history)}"
    history.append({"role": "assistant", "content": reply})
    score = evaluate_understanding(user_input, reply)

    print(f"[{user_id}] 👤 {user_input}")
    print(f"[{user_id}] 🤖 {reply} | 📊 Score: {score:.2f}")

    write_log(time.time(), user_id, user_input, reply, score)

    return jsonify({"text": reply})

#========自动续说检测===========
@app.route("/check_update", methods=["POST"])
def check_update():
    data = request.json
    user_id = data.get("user_id", "anonymous")

    if user_id not in session_memory:
        return jsonify({"update": False})

    session = session_memory[user_id]
    pending_reply = session.get("pending_auto_reply", None)

    if pending_reply:
        session["pending_auto_reply"] = None  # 清除已读
        return jsonify({"update": True, "text": pending_reply})
    else:
        return jsonify({"update": False})

# === 自动续说线程 ===
def auto_continue_check():
    while True:
        time.sleep(10)
        now = time.time()
        for uid, session in session_memory.items():
            last_time = session.get("last_active", now)
            if now - last_time > 30:
                history = session["history"]
                prompt = random.choice(AUTO_CONTINUE_TEMPLATES)
                history.append({
                    "role": "system",
                    "content": f"用户沉默了，请你以温柔朋友的语气继续说一些话，参考这条提示：{prompt}"
                })
                reply = get_reply(history)
                history.append({"role": "assistant", "content": reply})
                session["pending_auto_reply"] = reply
                write_log(time.time(), uid, "[auto_continue]", reply, "")

                session["last_active"] = now
                print(f"[{uid}] 🤖 自动续说：{reply}")

if __name__ == "__main__":
    threading.Thread(target=auto_continue_check, daemon=True).start()
    app.run(port=5005)

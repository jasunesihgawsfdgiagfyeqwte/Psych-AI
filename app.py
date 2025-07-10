import os
import json
import sys
import time
import random
import msvcrt
from dotenv import load_dotenv
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer
from utils.preprocess import load_esconv
from utils.vector_index import encode_text
import threading
import queue

os.chdir(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__))
# ========== AUTO_CONTINUE_TEMPLATES ==========
AUTO_CONTINUE_TEMPLATES = [
    "有时候一时说不上来也没关系，你觉得这个话题让你想到了什么？",
    "听起来这里面好像有些没说出口的东西？你方便说说看吗？",
    "我们可以不着急——你刚才那句话让我有点在想，你自己是怎么理解的？",
    "如果你愿意，我们可以继续沿着刚才那个话题聊下去。",
    "你刚刚停顿了一下，我猜你可能在想一些更深的事情？",
]
# ========== 加载环境变量 ==========
load_dotenv("API.env")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL")

# ========== 初始化客户端 ==========
client = OpenAI(api_key=MOONSHOT_API_KEY, base_url=MOONSHOT_BASE_URL)

# ========== 加载 FAISS 与语料 ==========
print("📦 正在加载向量索引与语料库...")
index_path = "Data/esconv_faiss.index"
json_path = "Data/ESConv.json"
faiss_index = faiss.read_index(index_path)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_pairs = load_esconv(json_path)

# ========== 读取 system prompt ==========
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = {"role": "system", "content": f.read()}

chat_history = [system_prompt]
context_injected = False
top_k = 3

# ========== 回复函数 ==========
def get_reply(prompt_messages, max_tokens=60):
    response = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=prompt_messages,
        temperature=0.8,
        timeout=20,
        response_format={"type": "json_object"},
        max_tokens=max_tokens
    )
    try:
        content = json.loads(response.choices[0].message.content)
        return content.get("text", "").strip()
    except json.JSONDecodeError:
        return response.choices[0].message.content.strip()

# ========== 输入超时函数 ==========
def safe_input_with_timeout(prompt, timeout):
    """
    完全非阻塞输入，并强制处理 ctrl-c 与空输入
    """
    input_queue = queue.Queue()

    def read_input(q):
        try:
            line = input()
            q.put(line.strip())
        except Exception:
            q.put("")

    print(prompt, end='', flush=True)
    thread = threading.Thread(target=read_input, args=(input_queue,), daemon=True)
    thread.start()
    thread.join(timeout)

    if not input_queue.empty():
        return input_queue.get()
    else:
        print()  # 自动换行
        return ""


# ========== 语料注入函数 ==========
def inject_context(user_text, chat_history, context_injected_flag):
    try:
        translation_response = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": "请将以下中文翻译为英文，不要输出任何其他内容。"},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
        )
        translated_query = translation_response.choices[0].message.content.strip()
        query_vector = encode_text(translated_query, embedding_model)
        D, I = faiss_index.search(query_vector, top_k)

        matched_contexts = []
        for idx in I[0]:
            if idx < len(corpus_pairs):
                entry = corpus_pairs[idx]
                matched_contexts.append(f"Q: {entry['question']}\nA: {entry['answer']}")

        if matched_contexts and not context_injected_flag:
            combined_context = "\n\n---\n\n".join(matched_contexts)
            chat_history.insert(1, {
                "role": "system",
                "content": f"以下是知识库背景信息，请在回答中参考但不要重复内容：\n\n{combined_context}"
            })
            return True
    except Exception as e:
        print("❌ 插入语料失败：", str(e))
    return context_injected_flag


while True:
    try:
        user_input = safe_input_with_timeout("你：", timeout=9999)

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            print("⚠️ 没有听到你的回应，要不你说点什么？")
            continue

        # === 上下文注入 + 回复 ===
        context_injected = inject_context(user_input, chat_history, context_injected)
        chat_history.append({"role": "user", "content": user_input})

        first_sentence = get_reply(chat_history, max_tokens=200)
        print("\n🤖 Kimi：", first_sentence)
        chat_history.append({"role": "assistant", "content": first_sentence})

        # === 沉默监听 + 自动续说 ===
        silent_rounds = 0
        while silent_rounds < 5:
            print("\n(你可以接着说，也可以沉默 20 秒让我继续)\n")
            follow_up = safe_input_with_timeout("你：", timeout=20)

            if follow_up.strip():
                # 用户继续说
                context_injected = inject_context(follow_up, chat_history, context_injected)
                chat_history.append({"role": "user", "content": follow_up})

                follow_reply = get_reply(chat_history, max_tokens=200)
                print("\n🤖 Kimi：", follow_reply)
                chat_history.append({"role": "assistant", "content": follow_reply})
                silent_rounds = 0  # 重置沉默计数器
            else:
                # 用户沉默，AI 主动继续说
                auto_prompt = random.choice(AUTO_CONTINUE_TEMPLATES)
                chat_history.append({
                    "role": "system",
                    "content": f"用户沉默了，请你以温柔朋友的语气继续说一些话，参考这条提示：{auto_prompt}"
                })

                continuation = get_reply(chat_history, max_tokens=200)
                if continuation.strip():
                    print("🤖 Kimi（继续）：", continuation)
                    chat_history.append({"role": "assistant", "content": continuation})
                silent_rounds += 1

    except Exception as e:
        print("❌ 出现错误：", str(e))
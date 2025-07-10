import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.preprocess import load_esconv    # 从你的预处理里复用函数
import os

def build_faiss_index():
    # 加载问答对
    file_path = os.path.join(os.path.dirname(__file__), "..", "Data", "ESConv.json")
    pairs = load_esconv(file_path)

    # 准备模型和问题
    model = SentenceTransformer("all-MiniLM-L6-v2")
    questions = [q for q, a in pairs]

    # 编码
    embeddings = model.encode(questions, show_progress_bar=True)

    # 构建索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # 保存索引
    faiss.write_index(index, "Data/esconv_faiss.index")
    print("✅ FAISS 索引已构建并保存！")

    return index, pairs, model

def encode_text(text, model):
    embedding = model.encode([text])
    return embedding.astype("float32")

if __name__ == "__main__":
    build_faiss_index()

import pandas as pd

# 指定输入输出文件名
input_file = "chat_logs.csv"
output_file = "chat_logs_archive_full.csv"

# 读取 CSV，指定列名和容错设置
df = pd.read_csv(
    input_file,
    header=None,
    names=["timestamp", "user_id", "user_input", "model_reply", "chat_type", "score"],
    quoting=1,
    encoding="utf-8",
    on_bad_lines="skip"
)

# 将 timestamp 转换为 datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

# 丢弃无法解析时间的行
df = df.dropna(subset=["timestamp"])

# 按说话人和时间排序
df = df.sort_values(by=["user_id", "timestamp"])

# 精简列并重命名
df = df[["timestamp", "user_id", "user_input", "model_reply"]].rename(columns={
    "timestamp": "时间",
    "user_id": "说话人",
    "user_input": "内容",
    "model_reply": "AI回应"
})

# 重置索引
df.reset_index(drop=True, inplace=True)

# 输出归档结果
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ 成功归档！共 {len(df)} 条记录，输出文件为：{output_file}")

import json
import os

def load_esconv(file_path):
    import json

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    pairs = []
    for dialog in raw_data:
        previous = None
        for turn in dialog["dialog"]:
            speaker = turn["speaker"]
            content = turn["content"]

            if speaker == "seeker":
                previous = content  # 保存 seeker 的问题
            elif speaker == "supporter" and previous:
                pairs.append({
                    "question": previous,
                    "answer": content
                })
                previous = None  # 匹配完清空

    return pairs

def save_moonshot_format(pairs, output_path="Data/moonshot_dataset.json"):
    output = []
    for question, answer in pairs:
        item = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        output.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

from flask import Flask, request, jsonify
from app import get_reply
import base64

app = Flask(__name__)

@app.route("/trtc_input", methods=["POST"])
def handle_input():
    data = request.json
    user_text = data.get("text", "")
    print("👂 收到：", user_text)

    reply = get_reply([{"role": "user", "content": user_text}], max_tokens=200)
    print("🤖 回复：", reply)

    filename = "trtc_reply.mp3"
    speak(reply, filename)

    with open(filename, "rb") as f:
        audio_data = f.read()

    return jsonify({
        "text": reply,
        "audio": base64.b64encode(audio_data).decode("utf-8")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)

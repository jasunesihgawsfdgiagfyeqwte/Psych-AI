# AI_PSYCH
 
# PsychAI: A Local LLM-Powered CBT Assistant with Contextual Memory

**PsychAI** is a lightweight, terminal-based mental health assistant powered by an LLM (via Moonshot API). It supports context injection from a fine-tuned emotional support dataset (ESConv), timeout-based continuation, and multi-turn memory—all running locally with minimal setup.

---

## 🚀 Features

- 🧠 LLM chat with warm, emotionally-aware replies
- 📚 Context retrieval from real support dialogues using FAISS & SentenceTransformers
- ⏳ Timeout-based auto continuation if user is silent
- 🔁 Multi-turn memory with System + User + Assistant roles
- 🔌 Fully local setup except API calls

---

## 📦 Requirements

```bash
pip install -r requirements.txt

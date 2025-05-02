# AI Resume Match & Career Assistant (RAG-powered)

This project is a **Retrieval-Augmented Generation (RAG)** application designed to help job seekers evaluate their resume against any job description.

🧠 Powered by:

* **FAISS** for fast semantic search
* **SentenceTransformers (MiniLM)** for embedding text chunks
* **Mistral-7B-Instruct** or **Llama-3.2-3b-Instruct** as the local language model for generation
* **Gradio** for an intuitive, browser-based interface

---

## 🚀 Live Demo (Hugging Face Space)

👉 \[[HF Resume RAG Assistant space](https://huggingface.co/spaces/BarrelRider/Resume_RAG_Assistant)]

---

## 🧩 Features

* 📄 **Upload Resume** (PDF) + **Paste Job Description**
* 🔍 Vectorize and index both using semantic chunking
* ❓ Ask questions like:

  * *"What skills am I missing?"*
  * *"Which job keywords are not in my resume?"*
  * *"Can you rewrite my summary to better fit this JD?"*
* 🤖 AI will retrieve relevant context and generate a smart, personalized answer

---

## 📦 Tech Stack

| Tool                    | Purpose                           |
| ----------------------- | --------------------------------- |
| `transformers`          | LLM inference (Mistral/Llama3.2)  |
| `sentence-transformers` | Embedding resume and JD chunks    |
| `faiss-cpu`             | Fast vector similarity search     |
| `gradio`                | Frontend interface                |
| `PyMuPDF`               | Resume PDF parsing                |

---

## 🛠 Installation (Locally)

```bash
git clone https://github.com/samarth003/resume-rag-assistant.git
cd resume-career-assistant
pip install -r requirements.txt
python app.py
```

---

## 📄 Requirements

**requirements.txt** includes:

```txt
gradio
transformers
sentence-transformers
faiss-cpu
torch
PyMuPDF
```
---

## 🔐 For Hugging Face Spaces Deployment

* Add your Hugging Face token as a secret: `HF_TOKEN`
* Make sure you have **agreed to model access** if using a gated model (like Mistral)

---

## 📬 Contact / Collaboration

Feel free to fork, test, or raise issues. PRs welcome!

---

## 📣 Credits

Created by [Samarth Kapoor](https://www.linkedin.com/in/kapoorsamarth/) — exploring AI for real-world productivity. 🚀
# 🏥 MediRAG – AI-Powered Medical Knowledge Assistant

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-00C7B7?style=for-the-badge&logo=render&logoColor=white)](https://medirag-ai.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Mistral AI](https://img.shields.io/badge/LLM-Mistral%20Large-FF7000?style=for-the-badge&logo=mistral&logoColor=white)](https://mistral.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**MediRAG** is an enterprise-grade, production-ready Medical RAG (Retrieval-Augmented Generation) application grounded in the 750-page **Gale Encyclopedia of Medicine** and custom user-uploaded medical records. 

By combining **BM25 Keyword Search** with **Chroma Vector Embeddings** and a **2nd-Stage FlashRank Cross-Encoder Reranker**, MediRAG eliminates hallucinations, delivers sub-second response times, and provides exact inline page citations.

🚀 **Live App**: [https://medirag-ai.onrender.com](https://medirag-ai.onrender.com)

---

## 🌟 Key Features

- **⚡ Advanced Hybrid Retrieval**: Combines **BM25 Sparse Keyword Search** (for exact medical terms like *"HbA1c"* or drug dosages) and **Chroma Dense Vector Search** (`BAAI/bge-small-en-v1.5`) via LangChain `EnsembleRetriever`.
- **🎯 2nd-Stage FlashRank Reranker**: Rescores retrieved candidate chunks using a local ONNX Cross-Encoder (`ms-marco-TinyBERT-L-2-v2`) to select the **top 3 hyper-relevant passages**.
- **🤖 Native Mistral AI Integration**: Powered by Mistral's flagship **`mistral-large-latest`** model for expert clinical reasoning and strict context grounding.
- **📖 Page-Level Source Citations**: Formats prompt context using `[Page XXX]` tags, enabling inline source attribution and transparent source expandable cards.
- **📄 Dynamic Custom PDF RAG**: Upload custom medical records/PDFs with instant indexing and session-state retriever caching.
- **🚨 Emergency Symptom Detection**: Real-time keyword filter detecting life-threatening queries (*chest pain, stroke, severe bleeding*) to display high-priority emergency alerts.
- **🎨 Modern SaaS UI**: Built with pure Streamlit CSS styling, responsive layout, dark medical design tokens, and empty-state welcome cards.
- **📥 Consultation Log Exporter**: One-click download of full chat history to `.txt` files for medical record keeping.

---

## 🏗️ System Architecture

```text
                  User Question / Custom PDF Upload
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  Dynamic Knowledge Base Router│
                 │ (Custom PDF Cache OR Gale DB) │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  1st-Stage Hybrid Search      │
                 │  • BM25 Keyword Search (k=8)  │
                 │  • Chroma Vector Search (k=8) │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  2nd-Stage FlashRank Reranker │
                 │  (Cross-Encoder -> Top 3)     │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  Grounded Prompt & Citations  │
                 │  (Injected [Page XXX] tags)   │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │ Mistral AI (mistral-large)    │
                 │ Direct Answer + Source Cards  │
                 └───────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Responsive SaaS web application |
| **LLM Engine** | Mistral AI (`mistral-large-latest`) | Clinical text generation & reasoning |
| **Embeddings** | Hugging Face (`BAAI/bge-small-en-v1.5`) | 384-dim dense vector embeddings |
| **Vector Store** | ChromaDB | Local persistent vector storage |
| **Keyword Search** | BM25 (`rank_bm25`) | Sparse keyword indexing |
| **Reranker** | FlashRank (`ms-marco-TinyBERT-L-2-v2`) | ONNX 2nd-stage cross-encoder reranking |
| **Document Processing** | PyPDFLoader + RecursiveTextSplitter | 700-token chunking with 120-token overlap |
| **Deployment** | Render.com + Docker | Cloud hosting with automated CI/CD |

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.12+
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/partu25/Parth-medical-bot.git
cd Parth-medical-bot
```

### 3. Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### 6. (Optional) Run Ingestion
To re-index the 750-page Gale Encyclopedia:
```bash
python ingest.py
```

### 7. Launch the Streamlit Web App
```bash
streamlit run web_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

You can build and run MediRAG locally using Docker:

### Build Container
```bash
docker build -t medirag-ai .
```

### Run Container
```bash
docker run -p 8501:10000 --env-file .env medirag-ai
```

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml          # Streamlit server & port configuration
├── medical_db/              # Pre-indexed ChromaDB vector store (750-page encyclopedia)
├── Dockerfile               # Docker configuration for Render deployment
├── ingest.py                # Document splitting & embedding script
├── web_app.py               # Main Streamlit web application & MediRAG UI
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API Keys)
└── README.md                # Project documentation
```

---

## ⚠️ Medical Disclaimer

MediRAG is designed strictly for **educational and informational purposes**. It does not provide medical diagnoses, treatment plans, or formal healthcare consultations. In case of a life-threatening medical emergency, call your local emergency services (911 / 112 / 108) immediately.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

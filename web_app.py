import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

load_dotenv()

st.set_page_config(
    page_title="MediRAG – AI-Powered Medical Knowledge Assistant", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    /* Main Background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Hide Default Header & Footer Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header Bar Component */
    .medirag-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05);
    }
    
    .medirag-title-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .medirag-icon-badge {
        background: linear-gradient(135deg, #0284C7 0%, #0EA5E9 100%);
        color: white;
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.2);
    }
    
    .medirag-title {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    
    .medirag-subtitle {
        font-size: 13px;
        font-weight: 500;
        color: #64748B;
        margin: 2px 0 0 0;
    }
    
    .medirag-status-badge {
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2);
    }

    /* Medical Disclaimer Banner */
    .medirag-disclaimer {
        background-color: #EFF6FF;
        border-left: 4px solid #0284C7;
        border-radius: 6px;
        padding: 10px 16px;
        color: #1E40AF;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Welcome Container */
    .welcome-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
    }
    
    .welcome-title {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    
    .welcome-text {
        font-size: 14px;
        color: #64748B;
        max-width: 580px;
        margin: 0 auto 24px auto;
        line-height: 1.6;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 20px;
        text-align: left;
    }
    
    .feature-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
    }
    
    .feature-card-icon {
        font-size: 20px;
        margin-bottom: 8px;
    }
    
    .feature-card-title {
        font-size: 14px;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 4px;
    }
    
    .feature-card-desc {
        font-size: 12px;
        color: #64748B;
        line-height: 1.4;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 24px;
    }
    
    .sidebar-section-header {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin: 16px 0 10px 0;
    }

    /* Custom Button Styling */
    .stButton > button {
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
        color: #334155;
        border-radius: 8px;
        font-weight: 500;
        font-size: 13px;
        padding: 8px 12px;
        transition: all 0.15s ease-in-out;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    .stButton > button:hover {
        border-color: #0284C7;
        color: #0284C7;
        background: #F0F9FF;
    }

    /* Chat Messages Styling */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    
    [data-testid="stChatMessage"][data-testimonial-user="true"] {
        background-color: #F0F9FF;
        border-color: #BAE6FD;
    }

    /* Expanders for Sources */
    .stExpander {
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        background-color: #F8FAFC !important;
        margin-top: 12px !important;
    }
    
    /* Input Box Styling */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar / Branding
st.markdown("""
<div class="medirag-header">
    <div class="medirag-title-box">
        <div class="medirag-icon-badge">🏥</div>
        <div>
            <h1 class="medirag-title">MediRAG</h1>
            <p class="medirag-subtitle">AI-Powered Medical Knowledge Assistant</p>
        </div>
    </div>
    <div class="medirag-status-badge">
        <span class="status-dot"></span> Assistant Ready
    </div>
</div>
""", unsafe_allow_html=True)

# Permanent Medical Disclaimer Banner
st.markdown("""
<div class="medirag-disclaimer">
    ⚠️ <span><b>Medical Disclaimer</b>: MediRAG is designed for educational & informational purposes only. It does not provide medical diagnosis or replace professional healthcare consultations.</span>
</div>
""", unsafe_allow_html=True)

EMERGENCY_KEYWORDS = [
    "chest pain", "shortness of breath", "severe bleeding", "heart attack",
    "stroke", "unconscious", "choking", "poisoning", "suicide", "head injury",
    "difficulty breathing", "seizure", "anaphylaxis"
]

@st.cache_resource
def load_models():
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=hf_token
    )
    
    # Official Mistral AI Client via ChatMistralAI
    llm = ChatMistralAI(
        model="mistral-large-latest",
        api_key=mistral_api_key,
        temperature=0.2
    )
    return embeddings, llm

embeddings_model, llm_model = load_models()

def build_hybrid_reranked_retriever(_vectorstore):
    """Builds Hybrid Search (BM25 + Vector) + 2nd-Stage FlashRank Reranker."""
    # 1. Vector Search Candidate Retriever (Top 8)
    chroma_retriever = _vectorstore.as_retriever(search_kwargs={"k": 8})
    
    # 2. BM25 Keyword Candidate Retriever (Top 8)
    store_data = _vectorstore.get()
    docs = [
        Document(page_content=t, metadata=m if m else {})
        for t, m in zip(store_data["documents"], store_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 8
    
    # 3. Hybrid Ensemble Retriever (50% BM25 + 50% Vector)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    
    # 4. FlashRank 2nd-Stage Cross-Encoder Reranker (Top 3)
    compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    return reranked_retriever

DOCUMENT_PROMPT = PromptTemplate(
    template="[Page {page}]: {page_content}",
    input_variables=["page", "page_content"]
)

template = """You are an expert medical assistant using the provided medical documents and encyclopedia.
Answer the user's question directly, clearly, and thoroughly using ONLY the retrieved context below.

INSTRUCTIONS:
1. Provide a direct, well-structured medical answer.
2. Cite the exact page numbers from the context inline for key facts (e.g. "[Page 436]").
3. Do NOT include meta-commentary like "No specific page citation available" or explanations about what is missing.

CONTEXT:
{context}

USER QUESTION: 
{question}

DIRECT MEDICAL ANSWER WITH PAGE CITATIONS:"""

CUSTOM_PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

with st.sidebar:
    st.markdown('<div class="sidebar-section-header">💡 Quick Sample Questions</div>', unsafe_allow_html=True)
    sample_queries = [
        "What are early symptoms of Type 2 Diabetes?",
        "What is the treatment for acute Bronchitis?",
        "What is the first aid for a severe burn?",
        "How can Hypertension be prevented?"
    ]
    
    for q in sample_queries:
        if st.button(f"👉 {q}", use_container_width=True):
            st.session_state["sample_prompt"] = q

    st.markdown("---")
    st.markdown('<div class="sidebar-section-header">📄 Upload Custom Documents</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a medical PDF", type="pdf")
    
    if uploaded_file:
        file_name = uploaded_file.name
        if st.session_state.get("custom_pdf_name") != file_name:
            with st.spinner(f"Processing and indexing '{file_name}'..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                splits = text_splitter.split_documents(docs)

                custom_vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings_model
                )
                custom_retriever = build_hybrid_reranked_retriever(custom_vectorstore)
                
                st.session_state["custom_pdf_name"] = file_name
                st.session_state["custom_retriever"] = custom_retriever

        st.success(f"📄 Active Knowledge Base: Uploaded Custom PDF ('{st.session_state['custom_pdf_name']}')")
        retriever = st.session_state.get("custom_retriever")
    else:
        # Clear custom PDF session state when un-uploaded
        if "custom_pdf_name" in st.session_state:
            st.session_state.pop("custom_pdf_name", None)
            st.session_state.pop("custom_retriever", None)

        if os.path.exists("./medical_db"):
            vectorstore = Chroma(persist_directory="./medical_db", embedding_function=embeddings_model)
            retriever = build_hybrid_reranked_retriever(vectorstore)
            st.info("📚 Active Knowledge Base: Gale Encyclopedia of Medicine")
        else:
            st.error("Encyclopedia (medical_db) not found. Please run ingest.py or upload a PDF.")
            retriever = None

    if st.session_state.get("messages"):
        st.markdown("---")
        st.markdown('<div class="sidebar-section-header">📥 Export Consultation Log</div>', unsafe_allow_html=True)
        chat_transcript = "\n\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="Download Consultation Log",
            data=chat_transcript,
            file_name="medquery_consultation_log.txt",
            mime="text/plain",
            use_container_width=True
        )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key="answer"
    )

if retriever:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={
            "prompt": CUSTOM_PROMPT,
            "document_prompt": DOCUMENT_PROMPT
        },
        return_source_documents=True
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Welcome Screen if no messages in chat
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-container">
            <h2 class="welcome-title">Welcome to MediRAG</h2>
            <p class="welcome-text">AI-powered medical information assistant grounded in your medical knowledge base. Ask any question below or select a quick sample question from the sidebar.</p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-card-icon">📚</div>
                    <div class="feature-card-title">Gale Encyclopedia</div>
                    <div class="feature-card-desc">Indexed 750-page comprehensive medical reference database.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-card-icon">⚡</div>
                    <div class="feature-card-title">Hybrid Search</div>
                    <div class="feature-card-desc">BM25 keyword search + Chroma vector search ensemble.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-card-icon">🎯</div>
                    <div class="feature-card-title">2nd-Stage Reranking</div>
                    <div class="feature-card-desc">FlashRank Cross-Encoder reranking for precision context.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-card-icon">📄</div>
                    <div class="feature-card-title">Custom Document RAG</div>
                    <div class="feature-card-desc">Upload custom medical PDFs to query your own records.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display past conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("is_emergency"):
                st.error("🚨 **EMERGENCY WARNING**: If you are experiencing a life-threatening medical emergency, call your local emergency services (911 / 112 / 108) immediately.")
            if message.get("sources"):
                with st.expander("📚 View Reranked Sources & Page Citations"):
                    for idx, doc in enumerate(message["sources"]):
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Source #{idx+1} (Page {page}):**")
                        st.caption(doc.page_content[:300] + "...")

    # Determine prompt from chat_input OR sample question buttons
    prompt = None
    text_prompt = st.chat_input("Ask a medical question...")

    if text_prompt:
        prompt = text_prompt
    elif st.session_state.get("sample_prompt"):
        prompt = st.session_state.pop("sample_prompt")

    if prompt:
        is_emergency = any(kw in prompt.lower() for kw in EMERGENCY_KEYWORDS)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if is_emergency:
                st.error("🚨 **EMERGENCY WARNING**: If you or someone else is experiencing a life-threatening medical emergency, please call emergency services (911 / 112 / 108) immediately before proceeding.")

            with st.spinner("Searching Medical Knowledge Base..."):
                try:
                    response = qa_chain.invoke({"question": prompt})
                    answer = response["answer"]
                    sources = response.get("source_documents", [])
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"⚠️ **Mistral API Error**: {e}")
                    st.stop()

                if sources:
                    with st.expander("📚 View Reranked Sources & Page Citations"):
                        for idx, doc in enumerate(sources):
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Source #{idx+1} (Page {page}):**")
                            st.caption(doc.page_content[:300] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "is_emergency": is_emergency
            })
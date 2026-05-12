import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Medical AI Assistant", layout="wide")
st.title("🏥 Medical RAG Assistant")
st.subheader("Query the Encyclopedia or upload your own records")

# --- Initialize Models (Persistent) ---
@st.cache_resource
def load_models():
    # Pass API Key explicitly from env just in case
    api_key = os.getenv("NVIDIA_API_KEY")
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
        api_key=api_key
    )
    llm = ChatNVIDIA(
        model="nvidia/nemotron-mini-4b-instruct", 
        api_key=api_key,
        temperature=0.2
    )
    return embeddings, llm

embeddings_model, llm_model = load_models()

# --- 1. SETUP PROMPT (Added this back) ---
template = """You are a professional medical assistant using the Gale Encyclopedia of Medicine (or provided medical documents). 
Use the following pieces of retrieved context to answer the user's question. 
If the answer is not in the context, politely say that the information is not in the encyclopedia.

CONTEXT:
{context}

USER QUESTION: 
{question}

HELPFUL MEDICAL ANSWER:"""

CUSTOM_PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("Upload Custom Documents")
    uploaded_file = st.file_uploader("Upload a medical PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing your document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # Create temporary DB
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
            st.success("Custom Document Loaded!")
            
            # Added search_kwargs to pull 3 relevant chunks
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        # DEFAULT: This connects to your first Gale document (medical_db)
        if os.path.exists("./medical_db"):
            vectorstore = Chroma(persist_directory="./medical_db", embedding_function=embeddings_model,collection_name="langchain")
            # Added search_kwargs here too
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            st.info("Using Gale Encyclopedia as knowledge base.")
        else:
            st.error("Encyclopedia (medical_db) not found. Please run ingest.py or upload a PDF.")
            retriever = None

# --- Chat Interface ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key="answer"
    )

if retriever:
    # Added the prompt to the chain logic
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}, # Added this
        return_source_documents=True
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a medical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the Encyclopedia..."):
                response = qa_chain.invoke({"question": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
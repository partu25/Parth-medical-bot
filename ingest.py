import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PDF_NAME = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND (4).pdf"
PERSIST_DIR = "./medical_db"

def run_ingestion():
    if not os.path.exists(PDF_NAME):
        print(f"Error: {PDF_NAME} not found in the directory.")
        return

    print(f"--- Loading {PDF_NAME} ---")
    loader = PyPDFLoader(PDF_NAME)
    pages = loader.load()

    print("--- Splitting text into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    docs = text_splitter.split_documents(pages)

    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=hf_token
    )

    print(f"--- Embedding {len(docs)} chunks in batches of 100 to prevent API timeouts ---")
    
    # Initialize Chroma store
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    # Ingest in small batches of 100 to avoid HuggingFace API timeouts
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        print(f"Ingesting batch {i // batch_size + 1} / {(len(docs) + batch_size - 1) // batch_size} (chunks {i} to {i + len(batch)})...")
        vector_db.add_documents(batch)
    
    print(f"--- Database successfully created in {PERSIST_DIR} ---")

if __name__ == "__main__":
    run_ingestion()
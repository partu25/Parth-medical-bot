import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# Load environment variables from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configuration
PDF_NAME = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND (4).pdf"
PERSIST_DIR = "./medical_db"

def run_ingestion():
    # 1. Load PDF
    if not os.path.exists(PDF_NAME):
        print(f"Error: {PDF_NAME} not found in the directory.")
        return

    print(f"--- Loading {PDF_NAME} ---")
    loader = PyPDFLoader(PDF_NAME)
    pages = loader.load()

    # 2. Split into chunks
    print("--- Splitting text into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(pages)

    # 3. Initialize Embeddings

    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=hf_token
    )

    # 4. Create and persist the Vector DB
    print(f"--- Embedding {len(docs)} chunks. This will take a few minutes on your Mac M5... ---")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    print(f"--- Database successfully created in {PERSIST_DIR} ---")

if __name__ == "__main__":
    run_ingestion()
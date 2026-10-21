import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
cf_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
cf_api_token = os.getenv("CLOUDFLARE_API_TOKEN")

llm = ChatMistralAI(
    model=os.getenv("MISTRAL_MODEL", "open-mistral-nemo"),
    api_key=mistral_api_key,
    temperature=0.2
)

if cf_account_id and cf_api_token:
    embeddings = CloudflareWorkersAIEmbeddings(
        account_id=cf_account_id,
        api_token=cf_api_token,
        model_name="@cf/baai/bge-small-en-v1.5"
    )
else:
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=hf_token
    )

if not os.path.exists("./medical_db"):
    print("Error: medical_db not found. Please run ingest.py first.")
    exit()

import chromadb
db_path = os.path.abspath("./medical_db")
client = chromadb.PersistentClient(path=db_path)
vector_db = Chroma(
    client=client,
    persist_directory=db_path, 
    embedding_function=embeddings
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer" 
)

template = """You are a professional medical assistant using the Gale Encyclopedia of Medicine. 
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

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
)

print("\n--- Medical RAG Assistant Ready ---")

while True:
    user_input = input("\nEnter your medical question (or type 'exit'): ")
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    try:
        print(f"\nSearching the Encyclopedia for: '{user_input}'...")
        
        response = qa_chain.invoke({"question": user_input})
        
        print(f"\nASSISTANT: {response['answer']}")

    except Exception as e:
        print(f"An error occurred: {e}")
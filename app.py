import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    api_key=api_key
)

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=hf_token
)

if not os.path.exists("./medical_db"):
    print("Error: medical_db not found. Please run ingest.py first.")
    exit()

vector_db = Chroma(
    persist_directory="./medical_db", 
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
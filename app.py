import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# Use langchain_classic just like you did in your original code
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

# Initialize Models
embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

llm = ChatNVIDIA(
    model="nvidia/nemotron-mini-4b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.2
)

# Load the existing DB created by ingest.py
if not os.path.exists("./medical_db"):
    print("Error: medical_db not found. Please run ingest.py first.")
    exit()

vector_db = Chroma(
    persist_directory="./medical_db", 
    embedding_function=embeddings
)

# 2. Setup Memory
# We specify output_key="answer" because ConversationalRetrievalChain returns the final response in the 'answer' key.
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer" 
)

# 3. Setup Prompt
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

# 4. Use ConversationalRetrievalChain instead of RetrievalQA
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
)

print("\n--- Medical RAG Assistant Ready ---")

# 5. Run the conversational loop
while True:
    user_input = input("\nEnter your medical question (or type 'exit'): ")
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    try:
        print(f"\nSearching the Encyclopedia for: '{user_input}'...")
        
        # Note: ConversationalRetrievalChain uses 'question' instead of 'query'
        response = qa_chain.invoke({"question": user_input})
        
        # The chain outputs the response under the 'answer' key
        print(f"\nASSISTANT: {response['answer']}")

    except Exception as e:
        print(f"An error occurred: {e}")
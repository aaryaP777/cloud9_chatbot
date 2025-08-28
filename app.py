from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from chatbot_engine import CohereEmbed
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3001"] for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load vector store
VECTOR_DB_PATH = "vector_store"
embedder = CohereEmbed()
db = FAISS.load_local(VECTOR_DB_PATH, embedder, allow_dangerous_deserialization=True)

# Prompt template for answering queries
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant helping with cloud optimization and security.

Context: {context}
Question: {question}

Answer the question based only on the context. If the answer is not found, say "I'm not sure about that."
"""
)

# Input model for API
class QueryRequest(BaseModel):
    question: str
    
@app.get("/")
def root():
    return {"message": "FastAPI backend is running!"}

@app.post("/chat")
async def chat(request: QueryRequest):
    if not request.question or not isinstance(request.question, str):
        raise HTTPException(
            status_code=400,
            detail="Request must include a 'question' field with a non-empty string"
        )
        
    query = request.question

    # Retrieve similar documents
    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Fill the prompt
    final_prompt = prompt.format(context=context, question=query)

    # Use Cohere's generate endpoint
    import cohere
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    response = co.generate(prompt=final_prompt, model="command-r-plus")

    return {
        "response": response.generations[0].text.strip(),
        "matched_docs": [doc.page_content for doc in docs]
    }

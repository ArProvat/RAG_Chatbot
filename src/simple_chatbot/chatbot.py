import os, json, pickle, pathlib, json5
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from ingest import parse_pdf_with_gemini
from chunker import build_docs, chunk_docs
from embed_and_index import build_or_load_index
from retriever import build_retriever
from generator import get_llm
from workflow import ChatState, build_workflow

# -----------------------------
# Retry wrapper for PDF parsing
# -----------------------------
@retry(
    stop=stop_after_attempt(5),                    
    wait=wait_exponential(multiplier=1, min=2, max=20),  
    retry=retry_if_exception_type(ServerError)    
)
def parse_pdf_retry(pdf_path: str):
    return parse_pdf_with_gemini(pdf_path)

# -----------------------------
# Load or parse PDF (with retry)
# -----------------------------
def load_or_parse_pdf(pdf_path: str, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    parsed_json_path = os.path.join(cache_dir, "parsed.json")
    chunks_path = os.path.join(cache_dir, "chunks.pkl")

    try:
        if os.path.exists(parsed_json_path) and os.path.exists(chunks_path):
            print("[Cache] Loading parsed JSON + chunks...")
            with open(parsed_json_path, "r", encoding="utf-8") as f:
                parsed_json = json5.load(f)
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
        else:
            print("[Ingest] Parsing PDF fresh ...")
            parsed_json = parse_pdf_retry(pdf_path)
            docs = build_docs(parsed_json)
            chunks = chunk_docs(docs)

            with open(parsed_json_path, "w", encoding="utf-8") as f:
                json5.dump(parsed_json, f, ensure_ascii=False, indent=2)
            with open(chunks_path, "wb") as f:
                pickle.dump(chunks, f)

    except Exception as e:
        print(f"[Error] {e}")
        return None, None

    return parsed_json, chunks

# -----------------------------
# Retrieval Function
# -----------------------------
def retrieval_func(state: ChatState):
    pdf_path = "Data/20200311-sitrep-51-covid-19.pdf"
    query = state["messages"][-1].content
    index_dir = "faiss_index"
    _, chunks = load_or_parse_pdf(pdf_path, cache_dir="cache")
    vectorstore = build_or_load_index(chunks, index_dir=index_dir)
    
   
    retriever = build_retriever(vectorstore, chunks)

    docs = retriever.invoke(query)
    print(f"[Retrieval] Found {len(docs)} relevant chunks.")
    return {"retrieval_docs": docs if docs else []}

# -----------------------------
# Generation Function
# -----------------------------
def generation_func(state: ChatState):
    query = state["messages"][-3]
    docs = state["retrieval_docs"]

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="""
You are a helpful assistant.
Answer ONLY from the provided context.
If not enough context, reply exactly:
"I don't know because I don't find relevant content yet."

Context:
{context}

User Question:
{query}

Answer:
"""
    )

    llm = get_llm("phi")
    response = llm.invoke(prompt.format(context=context, query=query))

    return {"messages": [AIMessage(content=response.content)]}





ChatBot = build_workflow(retrieval_func, generation_func)

'''
if __name__ == "__main__":

    config={"configurable": {"thread_id": "test_thread"}}
    for message_chunk, metadata in ChatBot.stream(
            {"messages": [HumanMessage(content='What is the total number of COVID-19 cases reported in japan ?')]},
            config=config,
            stream_mode="messages",
        ):
           print(message_chunk.content)'''
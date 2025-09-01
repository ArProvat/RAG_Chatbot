from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
import os , pickle

def build_or_load_index(chunks=[], index_dir="faiss_index"):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                  model_kwargs={'device': 'cpu'})
    if os.path.exists(index_dir):
        return FAISS.load_local(index_dir, embed, allow_dangerous_deserialization=True)
    vs = FAISS.from_documents(chunks, embed)
    lexical_retriever = BM25Retriever.from_documents(chunks)
    with open("lexical_retriever.pkl", "wb") as f:
        pickle.dump(lexical_retriever, f)
    vs.save_local(index_dir)
    return vs

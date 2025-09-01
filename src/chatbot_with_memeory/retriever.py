from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import pickle
def build_retriever(vectorstore, chunks):
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    with open("lexical_retriever.pkl", "rb") as f:
        lexical_retriever = pickle.load(f)
        
    ensemble = EnsembleRetriever(retrievers=[semantic_retriever,lexical_retriever], weights=[0.7, 0.3])
    return ensemble

''' model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base",
    model_kwargs={"device": "cpu"}
)
    compressor = CrossEncoderReranker(model=model, top_n=3)

    ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)
'''
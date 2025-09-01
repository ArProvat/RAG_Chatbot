from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def build_retriever(vectorstore, chunks):
    semantic = vectorstore.as_retriever(search_kwargs={"k": 5})
    lexical = BM25Retriever.from_documents(chunks)

    ensemble = EnsembleRetriever(retrievers=[semantic, lexical], weights=[0.7, 0.3])

    model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-base",
    model_kwargs={"device": "cpu"}
)
    compressor = CrossEncoderReranker(model=model, top_n=3)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

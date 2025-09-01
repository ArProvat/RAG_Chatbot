from langchain_ollama import ChatOllama

def get_llm(model="mistral"):
    return ChatOllama(model=model, temperature=0.7)

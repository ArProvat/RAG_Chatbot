from langchain_ollama import ChatOllama

def get_llm(model="mistral"):
    try:
       return ChatOllama(model=model, temperature=0.7)
    except Exception as e:
        print(f"[Error] {e}")
        print("[Info] Make sure you have the Ollama app running with the model downloaded.")
        return None
    
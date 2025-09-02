from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()
def get_llm(model="mistral"):
    try:
       ''' llm =HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            task="text-generation",
            model_kwargs={"temperature":0.7}
        )
        return ChatHuggingFace(llm=llm)'''
        # Ollama setup - make sure you have the Ollama app running with the model downloaded
        # to run ollama models ,you need to have ennough VRAM 
        # models  most of the time don't work due to lack of VRAM
       return ChatOllama(model=model, temperature=0.7)
    except Exception as e:
        print(f"[Error] {e}")
        print("[Info] Make sure you have the Ollama app running with the model downloaded.")
        return None
    
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieval_docs: List[str]


def build_workflow(retrieval_func, generation_func):
    graph = StateGraph(state_schema=ChatState)
    graph.add_node("retriever", retrieval_func)
    graph.add_node("generator", generation_func)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)

    #conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    #checkpointer = SqliteSaver(conn)
    #return graph.compile(checkpointer=checkpointer)
    return graph.compile()

'''def retrieval_all_thread():
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    all_thread =set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_thread)'''
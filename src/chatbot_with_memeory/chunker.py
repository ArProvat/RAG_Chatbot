from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def build_docs(parsed_json: dict):
    docs = [Document(page_content=parsed_json["text"], metadata={"type": "text"})]

    for t in parsed_json.get("tables", []):
        table_str = f"table(caption:{t['caption']}): {t['headers']} | Rows: {t['rows']}"
        docs.append(Document(page_content=table_str, metadata={"type":"table"}))

    for img in parsed_json.get("images", []):
        img_str = f"image(caption:{img['caption']}) summary:{img['content_summary']}"
        docs.append(Document(page_content=img_str, metadata={"type":"image"}))

    return docs

def chunk_docs(docs, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

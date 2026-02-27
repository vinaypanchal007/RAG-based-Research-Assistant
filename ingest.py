import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "../data"

all_docs = []

for filename in os.listdir(DATA_PATH):
    if filename.endswith(".pdf"):
        filepath = os.path.join(DATA_PATH, filename)
        loader = PyPDFLoader(filepath)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = filename

        all_docs.extend(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore")

print(f"Processed {len(all_docs)} pages from multiple PDFs successfully")
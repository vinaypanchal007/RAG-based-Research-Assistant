from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = Ollama(model="gemma:2b")

query = input("Ask a question: ")

docs = db.similarity_search(query, k=3)
context = "\n\n".join(doc.page_content for doc in docs)

prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)
print("\nAnswer:\n")
print(response)

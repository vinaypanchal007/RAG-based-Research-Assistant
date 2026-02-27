from rag_chain import get_rag_chain

qa = get_rag_chain()

while True:
    query = input("\nAsk a question (or type exit): ")
    if query.lower() == "exit":
        break

    result = qa(query)
    print("\nAnswer:\n", result["result"])
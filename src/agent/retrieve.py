from src.vanila.rag import retrieve


def retriever(state):
    """
    Retrieve documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retrieve(
        question=question, collection_name='test_rag', create_db=False)

    return {"documents": documents, "question": question}

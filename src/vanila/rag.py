from utils.emb import get_hf_embeddings
from utils.vector_db import create_qdrant_db, get_qdrant_db
from src.preprocessing.doc_preprocessing import document_process
from utils.llm import get_llama


def vanila_rag(question: str, create_db: False):
    # split document
    docs = document_process(file_path="data/achilles-tendon-rupture.pdf")

    # create vectorDB
    if create_db:
        vectorstore = create_qdrant_db(get_hf_embeddings(), docs=docs,
                                       collection_name='test_rag')

    else:
        vectorstore = get_qdrant_db(
            get_hf_embeddings(), collection_name="test_rag")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # retrieve docs
    documents = retriever.invoke(question)
    contexts = []

    for document in documents:
        contexts.append(document.page_content)

    print(f'context: {contexts}')

    # get answer
    response = get_llama().invoke({"context": contexts, "question": question})

    print(f'question: {question}')

    print(f'response: {response}')

    return response

import yaml
from utils.emb import get_hf_embeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from utils.vector_db import create_qdrant_db, get_qdrant_db
from src.preprocessing.doc_preprocessing import document_process
from utils.llm import chat_llama_cpp


with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

PDF_FILE_PATH = config['vanila_rag']['pdf_file_path']


def retrieve(question: str, collection_name: str = 'test_rag', create_db=False):

    if create_db:
        # split document
        docs = document_process(file_path=PDF_FILE_PATH)
        vectorstore = create_qdrant_db(get_hf_embeddings(), docs=docs,
                                       collection_name=collection_name)

    else:
        vectorstore = get_qdrant_db(
            get_hf_embeddings(), collection_name=collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # retrieve docs
    documents = retriever.invoke(question)

    return documents


def vanila_rag(question: str, collection_name: str = 'test_rag'):

    contexts = []

    documents = retrieve(
        question=question, collection_name=collection_name, create_db=False)

    for document in documents:
        contexts.append(document.page_content)

    print(f'context: {contexts}')
    print(f'question: {question}')

    # get answer
    prompt = hub.pull("rlm/rag-prompt")
    llm = chat_llama_cpp()

    # response = chat_llama_cpp().invoke(
    #     {"context": contexts, "question": question})

    response = prompt | llm | StrOutputParser()

    print(f'response: {response}')

    return response

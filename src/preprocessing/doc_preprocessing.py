from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def document_process(file_path: str):
    """
    process documents in local DB
    """
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)

    docs = text_splitter.split_documents(documents)

    print(f'docucment total page: {len(documents)}, total chunks: {len(docs)}')
    return docs

from langchain_qdrant import Qdrant


def create_qdrant_db(embeddings: object, docs: list, collection_name: str, location: str = "models/local_qdrant"):
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        path=location,
        collection_name=collection_name
    )

    return qdrant


def get_qdrant_db(embeddings: object, collection_name: str, location: str = "models/local_qdrant"):
    qdrant = Qdrant.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        path=location
    )

    return qdrant

# from qdrant_client import QdrantClient

# # Initialize the client
# # or QdrantClient(path="path/to/db")
# client = QdrantClient(path="models/local_qdrant")

# # Prepare your documents, metadata, and IDs
# docs = ["Qdrant has Langchain integrations",
#         "Qdrant also has Llama Index integrations"]
# metadata = [
#     {"source": "Langchain-docs"},
#     {"source": "Linkedin-docs"},
# ]
# ids = [42, 2]

# # Use the new add method
# client.add(
#     collection_name="test_rag",
#     documents=docs,
#     metadata=metadata,
#     ids=ids
# )

# search_result = client.query(
#     collection_name="test_rag",
#     query_text="This is a query document"
# )
# print(search_result)

from langchain_qdrant import Qdrant


def create_qdrant_db(embeddings: object, docs: list, collection_name: str, location: str = "models/local_qdrant"):
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        location=location,
        collection_name=collection_name
    )

    return qdrant


def get_qdrant_db(embeddings: object, collection_name: str, location: str = "models/local_qdrant"):
    qdrant = Qdrant.from_existing_collection(
        embeddings=embeddings,
        collection_name=collection_name,
        location=location
    )

    return qdrant

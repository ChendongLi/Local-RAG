from langchain_qdrant import Qdrant
import yaml

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

QDRANT_LOCATION = config['vanila_rag']['qdrant_disk_path']


def create_qdrant_db(embeddings: object, docs: list, collection_name: str, location: str = QDRANT_LOCATION):
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        path=location,
        collection_name=collection_name
    )

    return qdrant


def get_qdrant_db(embeddings: object, collection_name: str, location: str = QDRANT_LOCATION):
    qdrant = Qdrant.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        path=location
    )

    return qdrant

from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_hf_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    return embeddings


if __name__ == '__main__':
    embeddings = get_hf_embeddings()

    text = "This is a test document."

    query_result = embeddings.embed_query(text)

    print(len(query_result))

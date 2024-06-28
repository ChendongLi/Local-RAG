from langchain_community.llms import LlamaCpp


def get_llama():
    llm = LlamaCpp(
        model_path="./models/codellama-7b.Q4_0.gguf",
        temperature=0.75,
        max_tokens=2000,
        top_p=1
    )

    return llm


if __name__ == '__main__':

    llm = get_llama()

    question = """
    Question: What is the capital of France?
    """

    print(llm.invoke(question))

# Generate

from utils.llm import ollam_f, chat_llama_cpp
from langchain import hub
from langchain_core.output_parsers import StrOutputParser


def rag_chain():
    """
    generate answer using RAG
    """
    prompt = hub.pull("rlm/rag-prompt")
    llm = chat_llama_cpp()

    return prompt | llm | StrOutputParser()


def generate_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain().invoke(
        {"context": documents, "question": question})

    print(f'[generate_answer] generate_answer: {documents}')
    print(f'[generate_answer] generate_answer: {generation}')

    return {"documents": documents, "question": question, "generation": generation}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    generate = state["generate"]
    force_generate = state["force_generate"]

    if generate or force_generate:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate_answer"

    else:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"

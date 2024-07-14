# Retrieval Grader
# Hallucination Grader

from pprint import pprint
from langchain_core.pydantic_v1 import BaseModel, Field
from src.agent.chain import chain_creator
from utils.llm import chat_llama_cpp, ollama, ollam_f

# Retrieval Data model


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def retrieval_grader():
    """
    Grader for assessing relevance of a retrieved document to a user question.
    """
    # llm = ollam_f()
    llm = chat_llama_cpp()

    # Prompt
    # remove It does not need to be a stringent test. The goal is to filter out erroneous retrievals form the prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    human = "Retrieved document: \n\n {document} \n\n User question: {question}"

    return chain_creator(llm, system, human, GradeDocuments)


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    generate = state["generate"]
    force_generate = state["force_generate"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader().invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        # filtered_docs.append(d)

        print(f'[grade documents]: {documents}')

        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            generate = True
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            generate = False
            continue

    return {"documents": filtered_docs, "question": question, "generate": generate, "force_generate": force_generate}


# Hallucination Data model

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def hallucination_grader():
    """
    Grader for assessing hallucination in a generated answer.
    """
    llm = chat_llama_cpp()

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    human = "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"

    return chain_creator(llm, system, human, GradeHallucinations)


# Answer Grader


# Answer Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def answer_grader():
    """
    Grader for assessing whether an answer addresses / resolves a question.
    """
    llm = chat_llama_cpp()

    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    human = "User question: \n\n {question} \n\n LLM generation: {generation}"

    return chain_creator(llm, system, human, GradeHallucinations)


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    force_generate = state["force_generate"] if "force_generate" in state else False

    score = hallucination_grader().invoke(
        {"documents": documents, "generation": generation}
    )
    print('score', score.binary_score)
    grade = score.binary_score

    # Check hallucination
    if grade == "yes" or force_generate:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        print(
            f'hallucination grader question: {question} and generation: {generation}')
        score = answer_grader().invoke(
            {"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes" or force_generate:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# hallucination_grader = hallucination_prompt | structured_llm_grader
# hallucination_grader.invoke({"documents": docs, "generation": generation})

from typing_extensions import TypedDict
from typing import List

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agent.retrieve import retriever
from src.agent.router import route_question
from src.agent.grader import grade_documents, grade_generation_v_documents_and_question
from src.agent.generate import generate_answer, decide_to_generate
from src.agent.rewriter import transform_query


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        generate: boolean to generate answer or not
        force_generate: boolean to force generate answer from human intervention
    """

    question: str
    generation: str
    documents: List[str]
    generate: bool
    force_generate: bool


def save_graph(graph: object, graph_name: str):
    try:
        graph.get_graph().draw_mermaid_png(
            output_file_path=f'data/image/{graph_name}.png')
    except Exception as e:
        print(f"Error generate lang graph png {e}")


def build_self_rag_graph():
    workflow = StateGraph(GraphState)
    # Define the nodes
    workflow.add_node("retrieve", retriever)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate_answer", generate_answer)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate_answer": "generate_answer",
        },
    )
    workflow.add_edge("transform_query", "retrieve")

    workflow.add_conditional_edges(
        "generate_answer",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate_answer",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    memory = SqliteSaver.from_conn_string(":memory:")
    graph = workflow.compile(checkpointer=memory,
                             interrupt_after=["transform_query"])

    save_graph(graph, "self_rag_graph")

    return graph

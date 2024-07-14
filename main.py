# TODO structure output
from pprint import pprint
from utils.llm import chat_llama_cpp, ollama
from src.agent.chain import chain_creator
from src.agent.grader import retrieval_grader
from src.agent.graph import build_self_rag_graph
from src.vanila.rag import vanila_rag
from src.agent.generate import generate_answer


# response = vanila_rag(
#     question="what is achilles tendon rupture?", create_db=False)


# question = "what is achilles tendon rupture?"
# document = "Orthopaedics / Achilles tendon rupture: management and rehabilitation  \n \nAchilles tendon rupture: management and rehabilitation, July 2022 2 forced into an upward pointing position (dorsiflexion). Sometimes , the Achilles tendon is weak, \nmaking it more prone to rupture. This could be due to specific medical conditions , e.g. \nrheumatological conditions or medication combinations , such as steroids and certain antibiotics.  \nIt can also occur when there has been long term Achilles tendonitis. This is where the tendon \nbecomes swollen and painful and leads to small tears within the tendon. These tears cause the \ntendon to become increasingly weak and therefore more susceptible to r upture.  \n \nWhat are the s ymptoms ? \nWhen a rupture of the Achilles tendon occurs, you may experience a sudden pain in your heel or calf. The pain may then settle to a dull ache or it may go completely. This can be associated"

# score = retrieval_grader().invoke(
#     {"question": question, "document": document}
# )


# print(ollama())
# print(score)


def agent_graph_finite_loop(question: str):

    graph = build_self_rag_graph()

    inputs = {"question": question}
    thread = {"configurable": {"thread_id": "1"}}

    for output in graph.stream(inputs, thread):
        for key, value in output.items():
            pprint(f"[Node]: {key}")

    if len(graph.get_state(thread).next) > 0 and graph.get_state(thread).next[0] == 'retrieve':
        pprint("---FIRST INTERRUPT---")
        pprint(
            f"question: {graph.get_state(thread).values['question']}\nnext action: {graph.get_state(thread).next}")

        for output in graph.stream(None, thread):
            for key, value in output.items():
                pprint(f"[Node]: {key}")

    if len(graph.get_state(thread).next) > 0 and graph.get_state(thread).next[0] == 'retrieve':
        pprint("---SECOND INTERRUPT BEFORE UPDATE STATE---")
        pprint(
            f"question: {graph.get_state(thread).values['question']}\nnext action: {graph.get_state(thread).next}")

        current_state = graph.get_state(thread)
        current_state.values['force_generate'] = True
        graph.update_state(thread, current_state.values)

        pprint("---SECOND INTERRUPT AFTER UPDATE STATE---")
        pprint(
            f"question: {graph.get_state(thread).values['question']}\nnext action: {graph.get_state(thread).next}\nforece_generate: {graph.get_state(thread).values['force_generate']}")

        for output in graph.stream(None, thread):
            for key, value in output.items():
                pprint(f"[Node]: {key}")

        pprint(
            f"documents: {value['question']}\nquestion: {value['question']}\ngeneration: {value['generation']}")

    else:
        pprint(
            f"documents: {value['question']}\nquestion: {value['question']}\ngeneration: {value['generation']}")


if __name__ == "__main__":

    # agent_graph_finite_loop(
    #     "what is achilles tendon rupture?")

    agent_graph_finite_loop(
        "What player at the Bears expected to draft first in the 2024 NFL draft?")
    # response = vanila_rag(
    #     question="what is achilles tendon rupture?")

    # state = {"question": "what is achilles tendon rupture?", "documents": "Orthopaedics / Achilles tendon rupture: management and rehabilitation  \n \nAchilles tendon rupture: management and rehabilitation, July 2022 2 forced into an upward pointing position (dorsiflexion). Sometimes , the Achilles tendon is weak, \nmaking it more prone to rupture. This could be due to specific medical conditions , e.g. \nrheumatological conditions or medication combinations , such as steroids and certain antibiotics.  \nIt can also occur when there has been long term Achilles tendonitis. This is where the tendon \nbecomes swollen and painful and leads to small tears within the tendon. These tears cause the \ntendon to become increasingly weak and therefore more susceptible to r upture.  \n \nWhat are the s ymptoms ? \nWhen a rupture of the Achilles tendon occurs, you may experience a sudden pain in your heel or calf. The pain may then settle to a dull ache or it may go completely. This can be associated"
    #          }

    # result = generate_answer(state)

    # print(result)

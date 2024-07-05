from src.vanila.rag import vanila_rag


response = vanila_rag(
    question="what is achilles tendon rupture?", create_db=False)


# TODO structure output
# TODO create self correct agent

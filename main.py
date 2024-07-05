from src.vanila.rag import vanila_rag


response = vanila_rag(
    question="what is achilles tendon rupture?", create_db=False)


# TODO put hard code to config
# TODO structure output
# TODO create self correct agent

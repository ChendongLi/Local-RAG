import yaml
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser, SimpleJsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

LLAMA_MODEL_PATH = config['vanila_rag']['llama_cpp_model_path']


class RagOutputParser(BaseModel):
    """Output Format"""
    question: str = Field(
        default=None, description="the question that user asked")
    answer: str = Field(
        default=None, description="the answer that LLM generated")


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(
        default=None, description="A conversational response to the user's query")


def ollam_f():
    OllamaFunctions
    llm = OllamaFunctions(
        model="llama3",
        temperature=0,
        top_p=1,
        keep_alive=-1,
        format="json"
    )
    return llm


def ollama():
    """
    llama 3 from ollama
    """
    llm = ChatOllama(model="llama3", temperature=0,
                     top_p=1, stop=["<|eot_id|>"])

    return llm


def chat_llama_cpp():
    """
    llama cpp
    """
    llm = ChatLlamaCpp(
        model_path=LLAMA_MODEL_PATH,
        temperature=0,
        n_ctx=2000,
        max_tokens=1000,
        top_p=1,
        n_gpu_layers=-1,
        n_batch=512,
        verbose=False
    )

    return llm


def get_llama():
    system_prompt = (
        "You are AI assistant for question-answering task."
        "Use the following pieces of retrieved context to answer the question"
        "If you don't know the answer, say that you don't know. Keep the answer concise."
        "\n\n"
        "{context}"
    )

    #     "\n\n"
    # "Answer the question in the format below"
    # "{format_instructions}"

    # output_parser = PydanticOutputParser(
    #     pydantic_object=ConversationalResponse)

    template_messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        # MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)
    # prompt_template = ChatPromptTemplate.from_messages(template_messages).partial(
    #     format_instructions=output_parser.get_format_instructions())

    llm = LlamaCpp(
        model_path=LLAMA_MODEL_PATH,
        temperature=0,
        max_tokens=2000,
        top_p=1,
        n_gpu_layers=-1,
        n_batch=512,
        verbose=False
    )

    # memory = ConversationBufferMemory(
    #     memory_key="chat_history", return_messages=True)

    # chain = prompt_template | llm | output_parser
    chain = prompt_template | llm

    return chain


if __name__ == '__main__':

    chain = get_llama()

    context = """
    Don lives in North Vancouver BC
    """
    question = """
    Where Don lives?
    """

    print(chain.invoke({"context": context, "question": question}))

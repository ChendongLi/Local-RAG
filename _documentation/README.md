## llama-cpp-python installation

refer to https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md

(1) Make sure you have xcode installed... at least the command line parts

```
# check the path of your xcode install
xcode-select -p
```

(2) install a virtual environment using venv

```
python3 -m venv env_local_rag
```

(3) Install the LATEST llama-cpp-python...which happily supports MacOS Metal GPU as of version 0.1.62
Note: installation might be stuck there for a while "Building wheel for llama-cpp-python (pyproject.toml)"

```
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
pip install 'llama-cpp-python[server]'
```

(4) Download a v3 gguf v2 model. I use this model
https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/blob/main/codellama-7b.Q4_0.gguf

```
save gguf model under folder models/
```

## ollama installation

ollama is optimized version of llama cpp. Easy to install and fast response
refer to this [doc](https://github.com/ollama/ollama?tab=readme-ov-file)

(1) downlaod ollam
(2) run the following command to install llama3

```
ollama pull llama3
```

(3) run the ollama in python3

```
from langchain_community.chat_models import ChatOllama
def ollama():
    llm = ChatOllama(model="llama3")
    print(llm.invoke("Who is US president?"))
```

(4) [comparsion](https://picovoice.ai/blog/local-llms-llamacpp-ollama/) between llama_cpp with ollama

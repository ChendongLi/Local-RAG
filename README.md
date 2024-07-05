# Local-RAG

Build RAG using Local Llama-cpp and Qdrant in Disk

## Vanilla RAG Architecture

![image](data/vanilla-rag.png)

## Adaptive RAG Archtecture

![image](data/adaptive-rag.png)

## Repo Structure

```
Local-RAG
│   README.md
│   requirements.txt
│   .gitignore
│   main.py
└───_documentation (one time or manually operation)
│   │   README.md
└───config
└───data
└───models
└───secrets
└───src
│   └───preprocessing
        │   doc_preprocessing.py
│   └───vanila
│       │   rag.py
│   └───agent
└───utils
│   │   emb.py
│   │   llm.py
│   │   vector_db.py
```

## Install

```
pip install -r requirements.txt
```

## Usage / Getting started

```
python3 main.py
```

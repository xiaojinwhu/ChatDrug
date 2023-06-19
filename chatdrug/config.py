import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")

VECTORSTORE = os.getenv("VECTORSTORE")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "shibing624/text2vec-base-chinese"
)

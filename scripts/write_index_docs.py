import json
import os
from pathlib import Path
from uuid import uuid4

import tiktoken
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from tqdm.auto import tqdm

from chatdrug.config import VECTORSTORE
from chatdrug.embedding import embed
from chatdrug.utils import ChineseTextSplitter

tokenizer = tiktoken.get_encoding("p50k_base")


def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=128,
#     chunk_overlap=40,
#     length_function=tiktoken_len,
#     # separators=["。", "\n"],
# )

text_splitter = ChineseTextSplitter()

index_name = "langchain-chatdrug"


# loader = TextLoader("data/combined_data.txt")
# docs = loader.load()

data_dir = Path("data/txt")

texts = []
metadatas = []

for i in data_dir.iterdir():
    if i.suffix == ".txt":
        with open(i, "r", encoding="utf-8") as file:
            data = file.read()

        metadata = {"file_name": i.stem, "source": str(i)}

        splits_texts = text_splitter.split_text(data)
        chunks_metadatas = [
            {"chunk": i, "text": text, **metadata} for i, text in enumerate(data)
        ]
        texts.extend(splits_texts)
        metadatas.extend(chunks_metadatas)

vectordb = Chroma(index_name, embedding_function=embed, persist_directory=VECTORSTORE)

vectordb.add_texts(texts, metadata=metadatas)
vectordb.persist()
vectordb = None

import json
import os
from uuid import uuid4

import pinecone
import tiktoken
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

tokenizer = tiktoken.get_encoding("p50k_base")


def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=128,
    chunk_overlap=40,
    length_function=tiktoken_len,
    # separators=["。", "\n"],
)

model_name = "shibing624/text2vec-base-chinese"

embed = SentenceTransformerEmbeddings(model_name=model_name)

index_name = "langchain-chatdrug"
pinecone.init(
    api_key=os.getenv("PINECONE_TOKEN"), environment=os.getenv("PINECONE_LOC")
)

# pinecone.delete_index(index_name)
# pinecone.create_index(name=index_name, metric="cosine", dimension=768)

index = pinecone.Index(index_name)


batch_size = 128
texts = []
metadatas = []

with open("data/drug_list.json", "r") as f:
    data = json.load(f)

for i, d in enumerate(tqdm(data)):
    # print(record)
    metadata = {"drug_name": d["通用名称"], "source": d["通用名称"]}
    record = "\n".join([f"{k}：{str(v)}" for k, v in d.items()])
    records_texts = text_splitter.split_text(record)
    record_metadatas = [
        {"chunk": i, "text": text, **metadata} for i, text in enumerate(records_texts)
    ]
    texts.extend(records_texts)
    metadatas.extend(record_metadatas)

    if len(texts) >= batch_size:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []

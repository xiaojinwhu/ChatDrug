from langchain.embeddings import SentenceTransformerEmbeddings

from chatdrug.config import EMBEDDING_MODEL_NAME

embed = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

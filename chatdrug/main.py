from langchain import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma

from chatdrug.config import *
from chatdrug.embedding import embed
from chatdrug.llm import llm

temp = """
从下面内容中回答用户问题，如果出现类似本品等表述，需要找出其指代的药品。如果你不知道答案，就说你不知道，不要瞎编乱造。
{summaries}
{question}
"""

prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

                已知内容:
                {context}

                问题:
                {question}"""


class LLMQA:
    def __init__(self):
        self.llm = llm
        self.vectorstore = Chroma(
            persist_directory=VECTORSTORE, embedding_function=embed
        )
        self.qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=temp, input_variables=["summaries", "question"]
                )
            },
        )

    def ask(self, question):
        return self.qa(question)


if __name__ == "__main__":
    llmqa = LLMQA()
    print(llmqa.ask("糖尿病是什么"))

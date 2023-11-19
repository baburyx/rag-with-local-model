from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma


class VectorStoreHandler:
    def __init__(self) -> None:
        self.embedding = GPT4AllEmbeddings()  # type: ignore
        self.collection_name = "rag-private"
        pass

    def load_vectorstore(self, loaded_docs):
        vectorstore = Chroma.from_documents(
            documents=loaded_docs,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        return retriever

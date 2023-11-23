from langchain.embeddings import GPT4AllEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma


class VectorStoreHandler:
    def __init__(self) -> None:
        # self.embedding = GPT4AllEmbeddings()  # type: ignore
        self.model_name = "BAAI/bge-small-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        self.hf_embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )
        self.collection_name = "rag-private"
        pass

    def load_vectorstore(self, loaded_docs):
        vectorstore = Chroma.from_documents(
            documents=loaded_docs,
            collection_name=self.collection_name,
            embedding=self.hf_embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        return retriever

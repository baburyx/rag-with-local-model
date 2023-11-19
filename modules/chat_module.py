from langchain.chat_models import ChatOllama


class ChatHandler:
    def __init__(self) -> None:
        self.model_name = "llama2:7b-chat"
        self.chat_llm = ChatOllama(model=self.model_name)

    def ask_llm(self, query):
        """
        Sends a query to a chat model and returns its response.

        Parameters:
        - query: String query to send to the model.

        Returns:
        - Response content from the chat model.
        """
        response = self.chat_llm.invoke(query)
        return response.content

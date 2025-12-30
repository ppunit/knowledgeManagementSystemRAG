from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config import Config

class LLMService:
    def __init__(self,vector_store):
        self.llm = ChatGroq(
            model= "llama-3.1-8b-instant",
            temperature =0.7,
            groq_api_key= Config.GROQ_API_KEY
        )

        self.memory= ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm =self.llm,
            retriever = vector_store.vector_store.as_retriever(),
            memory=self.memory
        )

    def get_response(self, query):
        try:
            response = self.chain.run({"question":query})
            return response
        except Exception as e:
            return f"An error occurred: {str(e)}"

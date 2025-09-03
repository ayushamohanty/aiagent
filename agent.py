import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint


class NewsChat:
    def __init__(self, id: str):
        self.id = id

        # ✅ Use LangChain's built-in HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-large",
            max_new_tokens=512,
            temperature=0.2,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )

        # Prompt
        template = """You are a helpful AI assistant.
Answer the question based on the context below.

Context: {context}
Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        # Chroma retriever (adjust persist_directory as needed)
        retriever = Chroma(persist_directory="db").as_retriever()

        # Build RetrievalQA
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

    def ask(self, question: str) -> str:
        """Ask a question and get AI response."""
        response = self.rag_chain.invoke({"query": question})
        return response["result"]

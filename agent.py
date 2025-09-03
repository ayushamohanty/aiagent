import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import SentenceTransformerEmbeddings


class NewsChat:
    def __init__(self, id: str):
        self.id = id

        # ✅ HuggingFace LLM
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-large",
            max_new_tokens=512,
            temperature=0.2,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )

        # ✅ Embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # ✅ Prompt
        template = """You are a helpful AI assistant.
Answer the question based on the context below.

Context: {context}
Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        # ✅ Chroma retriever with embeddings
        retriever = Chroma(
            persist_directory="db",
            embedding_function=embeddings
        ).as_retriever()

        # ✅ RetrievalQA
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

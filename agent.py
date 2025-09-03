import os
from huggingface_hub import InferenceClient
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever


class HuggingFaceLLM:
    """Wrapper around HuggingFaceHub InferenceClient for LangChain compatibility."""

    def __init__(self, model_name="google/flan-t5-large", max_new_tokens=512, temperature=0.2):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("❌ Missing HUGGINGFACEHUB_API_TOKEN. Set it as env variable.")
        self.client = InferenceClient(model=model_name, token=token)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        response = self.client.text_generation(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response


class NewsChat:
    def __init__(self, id: str):
        self.id = id

        # Initialize custom LLM wrapper
        self.llm = HuggingFaceLLM(
            model_name="google/flan-t5-large",
            max_new_tokens=512,
            temperature=0.2,
        )

        # Simple prompt template
        template = """You are a helpful AI assistant.
Answer the question based on the context below.

Context: {context}
Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        # Load vector DB retriever (Chroma example)
        retriever: BaseRetriever = Chroma(persist_directory="db").as_retriever()

        # Build RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

    def ask(self, question: str) -> str:
        """Ask a question and get AI response."""
        response = self.rag_chain.invoke({"query": question})
        return response["result"]

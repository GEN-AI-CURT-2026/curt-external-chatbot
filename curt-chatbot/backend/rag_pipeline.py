import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pinecone import Pinecone
import cohere
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
import prompts

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "curt-external-chatbot-prod")


class CurtStructuredResponse(BaseModel):
    """The strict data schema returned to the React Frontend without suggestion arrays."""
    answer: str = Field(description="The finalized answering text or fallback message.")
    status: str = Field(description="The execution pipeline state flag.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Pipeline confidence alignment score.")


class CURTRagPipeline:
    def __init__(self):
        """
        Initialize the Highly Reliable RAG pipeline with built-in Retries, 
        Fallbacks, and Schema Guardrails.
        """
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise EnvironmentError("Missing required OPENAI_API_KEY for production.")

        self.primary_llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            request_timeout=30
        )

        self.fallback_llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125", 
            temperature=0,
            request_timeout=20
        )

        self.reliable_llm = (
            self.primary_llm
            .with_retry(
                retry_if_exception_type=(Exception,),
                stop_after_attempt=3,
                wait_exponential_jitter=True
            )
            .with_fallbacks(
                [self.fallback_llm],
                exceptions_to_handle=(Exception,)
            )
        )

        self.structured_parser_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        ).with_structured_output(CurtStructuredResponse)

        self.expansion_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "external-chatbot")
        self.index = self.pc.Index(self.index_name)
        self.namespace = "curt_docs"

        self.CONFIDENCE_HIGH = 0.80
        self.CONFIDENCE_LOW  = 0.30

        self._init_chains()

    def _init_chains(self):
        """Bind components using the reliable LLM runnables."""
        self.expansion_chain = (
            prompts.query_expansion_template | self.expansion_llm | StrOutputParser()
        )
        self.rag_chain = (
            prompts.rag_prompt_template | self.reliable_llm | StrOutputParser()
        )

    def _is_thank_you(self, text: str) -> bool:
        """Helper to scan user text for common conversational courtesies."""
        clean_text = text.lower().strip().replace(".", "").replace("!", "")
        patterns = [
            r"^(thank you|thanks|thx|ty|thank u)( mechanical)?( team)?$",
            r"^perfect thanks$",
            r"^great thanks$",
            r"^appreciate it$",
            r"^many thanks$"
        ]
        return any(re.match(p, clean_text) for p in patterns)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _query_pinecone_with_retry(self, query_vector: List[float], top_k: int = 10) -> List[Document]:
        """Queries Pinecone Vector Store with automatic exponential backoffs."""
        response = self.index.query(
            vector=query_vector, top_k=top_k, namespace=self.namespace, include_metadata=True
        )
        documents = []
        for match in response.matches:
            documents.append(Document(
                page_content=match.metadata.get("text", ""),
                metadata={
                    "source": match.metadata.get("source", "Unknown"),
                    "raw_score": match.score
                }
            ))
        return documents

    def run(self, query: str, chat_history: List[Dict] = []) -> Dict[str, Any]:
        """Runs free reasoning RAG logic, then passes results to schema validation."""

        if prompts.is_greeting(query):
            return {
                "answer": prompts.GREETING_RESPONSE, 
                "status": "greeting", 
                "confidence_score": 1.0
            }

        if self._is_thank_you(query):
            return {
                "answer": "You're very welcome! Dominate the track! 🏎️ Let me know if you need anything else down the line.",
                "status": "courtesy_response",
                "confidence_score": 1.0,
                "sources": []
            }

        expanded_query = self.expansion_chain.invoke({"query": query})

        query_vector = self.embeddings.embed_query(expanded_query)
        try:
            raw_docs = self._query_pinecone_with_retry(query_vector, top_k=10)
        except Exception:
            return {
                "answer": "Pipeline Error: The vector index is temporarily unreachable.",
                "status": "database_error", 
                "confidence_score": 0.0
            }

        if not raw_docs:
            return {
                "answer": prompts.NO_CONTEXT_RESPONSE, 
                "status": "no_context", 
                "confidence_score": 0.0
            }

        doc_texts = [doc.page_content for doc in raw_docs]
        rerank_response = self.cohere_client.rerank(
            model="rerank-english-v3.0", query=query, documents=doc_texts, top_n=5
        )
        valid_docs = [raw_docs[res.index] for res in rerank_response.results]

        primary_score = valid_docs[0].metadata.get("raw_score", 0.0)
        
        if primary_score < self.CONFIDENCE_LOW:
            return {
                "answer": prompts.NO_CONTEXT_RESPONSE, 
                "status": "fallback_insufficient_confidence",
                "confidence_score": primary_score
            }

        is_low_confidence = primary_score < self.CONFIDENCE_HIGH
        compressed_context = "\n\n".join([doc.page_content for doc in valid_docs])

        formatted_history = prompts.format_chat_history(chat_history)
        raw_rag_output = self.rag_chain.invoke({
            "context": compressed_context, "question": query, "chat_history": formatted_history
        })

        parsing_prompt = (
            f"Convert this free-form racing team chatbot answer into the required strict schema configuration.\n"
            f"If it mentions fallback scenarios, assign a status matching it.\n\n"
            f"Raw AI Answer:\n{raw_rag_output}"
        )
        
        try:
            structured_output: CurtStructuredResponse = self.structured_parser_llm.invoke(parsing_prompt)
            
            final_answer = structured_output.answer
            if is_low_confidence:
                final_answer = "*System Note: Displaying limited matching database data.*\n\n" + final_answer
                
            final_response = prompts.enhance_response_with_sources(final_answer, valid_docs)
            
            return {
                "answer": final_response,
                "status": "low_confidence_success" if is_low_confidence else structured_output.status,
                "confidence_score": primary_score,
                "sources": valid_docs
            }
        except Exception as validation_error:
            print(f"Schema Validation Guardrail caught parsing issue: {validation_error}")
            return {
                "answer": prompts.enhance_response_with_sources(raw_rag_output, valid_docs),
                "status": "unstructured_success",
                "confidence_score": primary_score
            }
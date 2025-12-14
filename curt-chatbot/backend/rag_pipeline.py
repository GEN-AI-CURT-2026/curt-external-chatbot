import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_core.output_parsers import StrOutputParser
import prompts
import build_chroma as chroma_config 
import cohere

# Load environment variables
load_dotenv()

class CURTRagPipeline:
    def __init__(self):
        """
        Initialize the RAG pipeline using configuration from build_chroma.py
        """
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        #query expansioin using prompts.py
        self.expansion_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        #for reranking
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

        #Load Vector Store and connect to chromadb
        db_path = str(chroma_config.CHROMA_DIR)
        model_name = chroma_config.EMBEDDING_MODEL
        collection_name = chroma_config.COLLECTION_NAME

        # print(f"Loading Vector Store from: {db_path}")
        # print(f"Using Embedding Model: {model_name}")
        
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})  #top 10 most relevant chunks

        self._init_chains()

    def _init_chains(self):
        """Initialize all LangChain runnables"""
        
        self.expansion_chain = (
            prompts.query_expansion_template  #from prompts.py
            | self.expansion_llm 
            | StrOutputParser()
        )

        self.compression_chain = (
            prompts.compression_template 
            | self.llm 
            | StrOutputParser()
        )

        self.rag_chain = (
            prompts.rag_prompt_template 
            | self.llm 
            | StrOutputParser()
        )

        self.hallucination_chain = (
            prompts.hallucination_check_template 
            | self.llm 
            | StrOutputParser()
        )
    def _rerank_with_cohere(self, query: str, documents: List, top_n: int = 5) -> List:
        """Rerank documents using Cohere's reranking API."""
        if not documents:
            return []
        
        doc_texts = [doc.page_content for doc in documents]
        
        try:
            rerank_response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=doc_texts,
                top_n=top_n,
                return_documents=True
            )
            
            reranked_docs = []
            for result in rerank_response.results:
                original_doc = documents[result.index]
                original_doc.metadata['rerank_score'] = result.relevance_score
                reranked_docs.append(original_doc)
            
            print(f"Cohere Reranking: {len(documents)} → {len(reranked_docs)} docs")
            
            return reranked_docs
            
        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            return documents[:top_n]
        
    def run(self, query: str, chat_history: List[Dict] = []) -> Dict[str, Any]:
        
        if prompts.is_greeting(query):
            return {"answer": prompts.GREETING_RESPONSE, "sources": [], "status": "greeting"}
        
        if prompts.is_off_topic(query):
            return {"answer": prompts.OFF_TOPIC_RESPONSE, "sources": [], "status": "off_topic"}

        expanded_query = self.expansion_chain.invoke({"query": query})
        print(f"Expanded Query: '{expanded_query}'")

        # Retrieval 
        raw_docs = self.retriever.invoke(expanded_query)
        #print(f"Retrieved {len(raw_docs)} raw documents")
        
        if not raw_docs:
            return {"answer": prompts.NO_CONTEXT_RESPONSE, "sources": [], "status": "no_docs"}

        #Reranking using cohere
        valid_docs = self._rerank_with_cohere(query, raw_docs, top_n=5)
        compressed_context = "\n\n".join([doc.page_content for doc in valid_docs])
            
        # Generation
        formatted_history = prompts.format_chat_history(chat_history)
        
        answer = self.rag_chain.invoke({
            "context": compressed_context,
            "question": query, 
            "chat_history": formatted_history
        })

        #Hallucination Detection
        check_result = self.hallucination_chain.invoke({
            "context": compressed_context,
            "answer": answer
        })
        
        print(f"Verification: {check_result}")

        if check_result.strip().upper().startswith("HALLUCINATION"):
            answer += "\n\n*(Note: I verified this answer against my database and found some parts might not be explicitly supported. Please verify with official team documents.)*"

        final_response = prompts.enhance_response_with_sources(answer, valid_docs)
        return {
            "answer": final_response,
            "raw_answer": answer,
            "sources": valid_docs,
            "expanded_query": expanded_query,
            "status": "success"
        }

def take_input(input):
    """Function to take input """
    return input

if __name__ == "__main__":
    try:
        pipeline = CURTRagPipeline()
        user_input = take_input("What are the current projects of the Cairo University Racing Team?")
        result = pipeline.run(user_input)
        print("\n" + "="*50)
        print("AI RESPONSE:")
        print(result["answer"])
        print("="*50)
        print("\nSPECIFIC CHUNKS USED:")
        
        if result["sources"]:
            for i, doc in enumerate(result["sources"], 1):
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                
                print(f"\n[Chunk {i}] Source: {source_name}")
                print(f"Content: {doc.page_content}")
                print("-" * 40)
        else:
            print("No relevant chunks were found.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure you have run 'python build_chroma.py' first!")
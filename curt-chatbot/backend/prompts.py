from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

#Pre-retrieval using query expansion
#query expansio rewrites vague queries to be more specific 
query_expansion_template = PromptTemplate(
    input_variables=["query"],
    template="""You are a query expansion assistant for the Cairo University Racing Team (CURT) chatbot.

Your task: Rewrite the user's question to be more specific and searchable while maintaining the original intent.

Guidelines:
- Expand abbreviations for example: "team" is expanded into "CURT team members, structure, roles")
- Add relevant context keywords
- Keep it concise (1-2 sentences maximum)
- If the query is already specific, return it unchanged


User Question: {query}

Expanded Query:"""
)


#Main RAG Prompt Template 
rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are the official Cairo University Racing Team (CURT) chatbot assistant. Your role is to provide accurate, helpful information about CURT based ONLY on the provided context.

Guidelines:
1. Answer ONLY using information from the provided context below, If you are unsure, say so.
2. If the context doesn't contain the answer, respond with: "I don't have that information in my knowledge base. You can contact CURT directly for more details."
3. NEVER make up or infer information not explicitly stated in the context
4. Be friendly, professional, and enthusiastic about CURT
5. If you mention achievements, dates, or specific facts, ensure they come directly from the context
6. Use conversation history to maintain context, but don't make assumptions

Context from CURT Knowledge Base:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}")
])


#Post-retrieval using Reranking 
#Reranking re orders chunks based on relevance to the query
compression_template = PromptTemplate(
    input_variables=["query", "chunk_text"],
    template="""Extract ONLY the sentences from the text below that are relevant to answering the question. 

If no sentences are relevant, respond with "NO_RELEVANT_CONTENT".

Question: {query}

Text:
{chunk_text}

Relevant Sentences (preserve original wording):"""
)


# Post-RetrievalL: Hallucination Detection

hallucination_check_template = PromptTemplate(
    input_variables=["context", "answer"],
    template="""Your task is to verify if an answer is grounded in the provided context.

Context:
{context}

Answer to Check:
{answer}

Question: Does the answer contain ANY information that is not present or cannot be directly inferred from the context?

Respond with ONLY:
- "GROUNDED" if the answer is fully supported by the context
- "HALLUCINATION: [specific issue]" if the answer contains unsupported information

Your Response:"""
)

#Basic Responses
NO_CONTEXT_RESPONSE = """I don't have specific information about that in my current knowledge base. 

Here's what I can help you with:
- CURT team structure and members
- History and achievements
- Current projects and competitions
- FAQs about joining or collaborating

You can also reach out to CURT directly through their official channels for more detailed information."""

GREETING_RESPONSE = """Hello! I'm the CURT chatbot, here to help you learn about the Cairo University Racing Team.

I can tell you about:
- Our team structure and members
- Competition history and achievements  
- Current projects
- How to get involved

What would you like to know?"""

OFF_TOPIC_RESPONSE = """I'm specifically designed to answer questions about CURT (Cairo University Racing Team). 

I can help you with information about:
- The team and its members
- Competitions and achievements
- Projects and technical work
- How to join or collaborate

Is there anything about CURT I can help you with?"""



def is_greeting(query: str) -> bool:
    """Check if query is a greeting."""
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    query_lower = query.lower().strip()
    return query_lower in greetings or any(query_lower.startswith(g + ' ') or query_lower.startswith(g + ',') for g in greetings)


def is_off_topic(query: str) -> bool:
    """
    Check if query is off-topic (not about CURT).
    Simple heuristic - can be improved with classification.
    """
    off_topic_keywords = [
        'weather', 'recipe', 'movie', 'sports score', 'politics', 
        'stock', 'celebrity', 'video game', 'restaurant', 'fashion'
    ]
    query_lower = query.lower()
    
    # If explicitly mentions CURT, it's on topic
    if any(term in query_lower for term in ['curt', 'car', 'racing team', 'competition','racing','cairo university']):
        return False
    
    # Check for off topic keywords
    return any(keyword in query_lower for keyword in off_topic_keywords)


def format_sources(chunks: List[Dict]) -> str:
    """
    Format source chunks for citation in the response.
    
    Args:
        chunks: List of retrieved chunks with metadata
        
    Returns:
        Formatted source string
    """
    if not chunks:
        return ""
    
    sources = "\n\n**Sources:**\n"
    seen_sources = set()
    
    for i, chunk in enumerate(chunks, 1):
        # Handle both Document objects and dict formats
        if hasattr(chunk, 'metadata'):
            source = chunk.metadata.get('source', 'Unknown')
        else:
            source = chunk.get('source', chunk.get('metadata', {}).get('source', 'Unknown'))
        
        # Avoid duplicate sources
        if source not in seen_sources:
            sources += f"{i}. {source}\n"
            seen_sources.add(source)
    
    return sources


def format_chat_history(messages: List[Dict]) -> List:
    """
    Convert message history to LangChain message format.
    
    Args:
        messages: List of dicts with 'role' and 'content'
        
    Returns:
        List of LangChain message objects
    """
    formatted = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'user':
            formatted.append(HumanMessage(content=content))
        elif role == 'assistant':
            formatted.append(AIMessage(content=content)) 
    
    return formatted


def enhance_response_with_sources(answer: str, chunks: List) -> str:
    """
    Add source citations to the generated answer.
    
    Args:
        answer: Generated answer from LLM
        chunks: Source chunks used (Document objects or dicts)
        
    Returns:
        Answer with appended sources
    """
    sources = format_sources(chunks)
    return f"{answer}\n{sources}" if sources else answer


from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import OLLAMA_HOST, OLLAMA_MODEL
from backend.data_loader import get_scholarship_by_id
from backend.rag.retriever import retrieve_policy

POLICY_QA_PROMPT_TEMPLATE = """You are an authoritative and student-friendly AI assistant specializing in college scholarship policies.
Answer the student's question accurately and concisely based strictly on the provided official policy context below.

Scholarship: {scholarship_name} (ID: {scholarship_id})

Official Policy Context:
{policy_context}

Student Question:
{question}

Instructions:
1. Provide a direct, clear, and helpful answer grounded ONLY in the policy context above.
2. If the policy context does not contain the answer, state politely that the provided policy text does not specify this information.
3. Keep the tone encouraging, professional, and clear for a college student."""

policy_qa_prompt = PromptTemplate(
    input_variables=["scholarship_id", "scholarship_name", "policy_context", "question"],
    template=POLICY_QA_PROMPT_TEMPLATE,
)

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_HOST,
    temperature=0.2,
)

policy_qa_chain = policy_qa_prompt | llm | StrOutputParser()


def answer_policy_question(scholarship_id: str, question: str) -> Optional[str]:
    """
    Retrieves policy context via RAG and answers the student's question
    grounded in the official policy documents using LangChain.
    Returns None if the scholarship ID is not found.
    """
    scholarship = get_scholarship_by_id(scholarship_id)
    if not scholarship:
        return None

    policy_context = retrieve_policy(query=question, scholarship_id=scholarship.id)

    result = policy_qa_chain.invoke({
        "scholarship_id": scholarship.id,
        "scholarship_name": scholarship.name,
        "policy_context": policy_context,
        "question": question,
    })

    return result.strip()

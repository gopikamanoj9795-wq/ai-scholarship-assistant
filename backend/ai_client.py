from typing import Optional
from backend.data_loader import get_scholarship_by_id
from backend.chains.summary_chain import summary_chain


def generate_summary(scholarship_id: str) -> Optional[str]:
    """
    Loads the scholarship by ID, invokes the LangChain summary_chain
    to explain the scholarship in one plain-English sentence for a student,
    and returns the cleaned response text. Returns None if the scholarship is not found.
    """
    scholarship = get_scholarship_by_id(scholarship_id)
    if not scholarship:
        return None

    docs_str = ", ".join(scholarship.required_docs) if scholarship.required_docs else "None"
    years_str = ", ".join(map(str, scholarship.eligible_years)) if scholarship.eligible_years else "All"

    result = summary_chain.invoke({
        "name": scholarship.name,
        "category": scholarship.category,
        "min_cgpa": scholarship.min_cgpa,
        "max_family_income": scholarship.max_family_income,
        "eligible_years": years_str,
        "amount": scholarship.amount,
        "deadline": scholarship.deadline,
        "required_docs": docs_str,
    })

    return result.strip()


def answer_policy_question(scholarship_id: str, question: str) -> Optional[str]:
    """
    Retrieves policy context via RAG and answers the student's question
    grounded in the official policy documents using LangChain.
    Returns None if the scholarship ID is not found.
    """
    from backend.chains.policy_qa_chain import answer_policy_question as _answer_policy_question
    return _answer_policy_question(scholarship_id, question)



from typing import List, Any, Dict
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import asdict

from backend.config import APP_NAME, APP_VERSION, APP_DESCRIPTION
from backend.data_loader import (
    load_students,
    load_scholarships,
    get_student_by_roll,
    match_scholarships_for_student,
)
from backend.ai_client import generate_summary, answer_policy_question
from backend.graph.workflow import run_scholarship_workflow


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/students", tags=["Students"])
def get_students() -> List[Dict[str, Any]]:
    """Loads and returns all students from data/students.json."""
    students = load_students()
    return [asdict(student) for student in students]


@app.get("/scholarships", tags=["Scholarships"])
def get_scholarships() -> List[Dict[str, Any]]:
    """Loads and returns all scholarships from data/scholarships.json."""
    scholarships = load_scholarships()
    return [asdict(scholarship) for scholarship in scholarships]


@app.get("/match/{roll_no}", tags=["Matching"])
def match_scholarships(roll_no: str) -> Dict[str, Any]:
    """
    Finds a student by roll_no and returns matching scholarships
    using the Scholarship.matches() method.
    """
    student = get_student_by_roll(roll_no)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with roll number '{roll_no}' was not found.",
        )

    matched = match_scholarships_for_student(student)
    return {
        "student": asdict(student),
        "total_matched": len(matched),
        "matches": [asdict(sch) for sch in matched],
    }


@app.get("/summary/{scholarship_id}", tags=["AI Summary"])
def get_scholarship_summary(scholarship_id: str) -> Dict[str, Any]:
    """
    Generates a concise one-sentence plain-English summary for a scholarship
    using LangChain and Ollama.
    """
    summary = generate_summary(scholarship_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scholarship with ID '{scholarship_id}' was not found.",
        )
    return {
        "scholarship_id": scholarship_id,
        "summary": summary,
    }


@app.get("/policy-question/{scholarship_id}", tags=["Policy RAG"])
def ask_policy_question(scholarship_id: str, question: str) -> Dict[str, Any]:
    """
    Answers questions about a scholarship's official policy using RAG
    grounded in the Chroma vector database and LangChain.
    """
    answer = answer_policy_question(scholarship_id=scholarship_id, question=question)
    if answer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scholarship with ID '{scholarship_id}' was not found.",
        )
    return {
        "scholarship_id": scholarship_id,
        "question": question,
        "answer": answer,
    }


@app.get("/workflow/{roll_no}", tags=["Workflow"])
def execute_workflow(roll_no: str) -> Dict[str, Any]:
    """
    Executes the full LangGraph stateful scholarship workflow for a given student roll number
    and returns the complete resulting state as JSON.
    """
    return run_scholarship_workflow(roll_no)


@app.get("/crew-workflow/{roll_no}", tags=["Crew Workflow"])
def execute_crew_workflow(roll_no: str) -> Dict[str, Any]:
    """
    Executes the full LangGraph + CrewAI stateful scholarship workflow
    combining Eligibility, Application Tracker, and Deadline Alert agents.
    """
    return run_scholarship_workflow(roll_no)





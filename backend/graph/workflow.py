import sys
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langgraph.graph import StateGraph, START, END

from backend.data_loader import get_student_by_roll
from backend.agents.crew import run_scholarship_crew
from backend.graph.state import AppState


def load_student(state: AppState) -> Dict[str, Any]:
    """
    Looks up the student by roll_no using data_loader.
    If not found, flags the state for human review.
    """
    roll_no = state.get("roll_no", "").strip()
    student = get_student_by_roll(roll_no)
    if not student:
        return {
            "student": None,
            "needs_human_review": True,
            "matched_scholarships": [],
            "applications": [],
            "urgent_deadlines": [],
        }

    return {
        "student": asdict(student),
        "needs_human_review": False,
    }


def human_review_node(state: AppState) -> Dict[str, Any]:
    """
    Handles cases requiring manual human intervention (e.g. unknown student ID).
    """
    roll_no = state.get("roll_no", "")
    message = f"Student with roll number '{roll_no}' was not found in the records. Flagged for manual human review."
    return {
        "final_response": message,
    }


def run_crew(state: AppState) -> Dict[str, Any]:
    """
    Runs the multi-agent CrewAI scholarship crew (Eligibility, Application Tracker,
    and Deadline Alert agents) and updates the workflow state with the results.
    """
    roll_no = state.get("roll_no", "").strip()
    crew_results = run_scholarship_crew(roll_no)

    return {
        "matched_scholarships": crew_results.get("matched_scholarships", []),
        "applications": crew_results.get("applications", []),
        "urgent_deadlines": crew_results.get("urgent_deadlines", []),
        "crew_output": crew_results.get("crew_output", ""),
    }


def build_response(state: AppState) -> Dict[str, Any]:
    """
    Combines student information, matched scholarships, active applications,
    and deadline alerts synthesized by the crew into a final response.
    """
    student_dict = state.get("student", {}) or {}
    student_name = student_dict.get("name", "Student")
    roll_no = state.get("roll_no", "")
    matched = state.get("matched_scholarships", [])
    apps = state.get("applications", [])
    urgent = state.get("urgent_deadlines", [])
    crew_output = state.get("crew_output", "")

    lines = [f"### Scholarship & Application Summary for {student_name} (Roll No: {roll_no})"]

    if not matched:
        lines.append("\n- **Eligibility**: No matching scholarships found based on your current profile.")
    else:
        sch_lines = [
            f"  * {s.get('name')} (ID: {s.get('id')}) - Award: INR {s.get('amount', 0):,.0f}, Deadline: {s.get('deadline')}"
            for s in matched
        ]
        lines.append(f"\n- **Eligible Scholarships ({len(matched)})**:\n" + "\n".join(sch_lines))

    if apps:
        app_lines = [
            f"  * Application {a.get('application_id')} for {a.get('scholarship_id')} - Status: {a.get('status')} (Applied: {a.get('applied_date')})"
            for a in apps
        ]
        lines.append(f"\n- **Active Applications ({len(apps)})**:\n" + "\n".join(app_lines))
    else:
        lines.append("\n- **Active Applications**: No previous scholarship applications submitted yet.")

    if urgent:
        urgent_lines = [
            f"  * {u.get('name')} (Deadline: {u.get('deadline')}) - Closing in {u.get('days_remaining', 0)} day(s)!"
            for u in urgent
        ]
        lines.append(f"\n- **URGENT DEADLINES ({len(urgent)})**:\n" + "\n".join(urgent_lines))
    else:
        lines.append("\n- **Deadlines**: All deadlines are on schedule; none closing within the next 5 days.")

    if crew_output:
        lines.append(f"\n- **AI Agent Guidance**:\n{crew_output}")

    return {
        "final_response": "\n".join(lines),
    }


def route_after_load(state: AppState) -> str:
    """Conditional router: skips to human review if student not found."""
    if state.get("needs_human_review", False):
        return "human_review"
    return "run_crew"


# Construct the StateGraph
workflow_builder = StateGraph(AppState)

workflow_builder.add_node("load_student", load_student)
workflow_builder.add_node("human_review", human_review_node)
workflow_builder.add_node("run_crew", run_crew)
workflow_builder.add_node("build_response", build_response)

# Connect edges
workflow_builder.add_edge(START, "load_student")
workflow_builder.add_conditional_edges(
    "load_student",
    route_after_load,
    {
        "human_review": "human_review",
        "run_crew": "run_crew",
    },
)
workflow_builder.add_edge("human_review", END)
workflow_builder.add_edge("run_crew", "build_response")
workflow_builder.add_edge("build_response", END)

# Compile workflow
scholarship_workflow = workflow_builder.compile()


def run_scholarship_workflow(roll_no: str) -> Dict[str, Any]:
    """
    Executes the compiled LangGraph + CrewAI scholarship workflow
    and returns the resulting state.
    """
    initial_state: AppState = {
        "roll_no": roll_no,
        "student": None,
        "matched_scholarships": [],
        "applications": [],
        "urgent_deadlines": [],
        "crew_output": None,
        "needs_human_review": False,
        "final_response": "",
    }
    return scholarship_workflow.invoke(initial_state)

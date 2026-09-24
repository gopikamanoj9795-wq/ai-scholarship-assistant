from typing import TypedDict, Optional, List, Dict, Any


class AppState(TypedDict, total=False):
    """
    Represents the complete state of the scholarship workflow pipeline.
    """
    roll_no: str
    student: Optional[Dict[str, Any]]
    matched_scholarships: List[Dict[str, Any]]
    applications: List[Dict[str, Any]]
    urgent_deadlines: List[Dict[str, Any]]
    crew_output: Optional[str]
    needs_human_review: bool
    final_response: str

import json
from typing import List, Optional, Dict, Any
from models.student import Student
from models.scholarship import Scholarship
from backend.config import STUDENTS_FILE, SCHOLARSHIPS_FILE, APPLICATIONS_FILE


def load_students() -> List[Student]:
    """Load all students from data/students.json."""
    if not STUDENTS_FILE.exists():
        return []
    with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Student(**item) for item in data]


def load_scholarships() -> List[Scholarship]:
    """Load all scholarships from data/scholarships.json."""
    if not SCHOLARSHIPS_FILE.exists():
        return []
    with open(SCHOLARSHIPS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Scholarship(**item) for item in data]


def load_applications() -> List[Dict[str, Any]]:
    """Load all applications from data/applications.json."""
    if not APPLICATIONS_FILE.exists():
        return []
    with open(APPLICATIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_applications_by_roll(roll_no: str) -> List[Dict[str, Any]]:
    """Find all applications for a student by roll number (case-insensitive)."""
    apps = load_applications()
    roll_clean = roll_no.strip().lower()
    return [a for a in apps if a.get("roll_no", "").strip().lower() == roll_clean]


def get_student_by_roll(roll_no: str) -> Optional[Student]:
    """Find a student by roll number (case-insensitive)."""
    students = load_students()
    roll_clean = roll_no.strip().lower()
    for student in students:
        if student.roll_no.strip().lower() == roll_clean:
            return student
    return None


def get_scholarship_by_id(scholarship_id: str) -> Optional[Scholarship]:
    """Find a scholarship by ID (case-insensitive)."""
    scholarships = load_scholarships()
    sch_clean = scholarship_id.strip().lower()
    for sch in scholarships:
        if sch.id.strip().lower() == sch_clean:
            return sch
    return None


def match_scholarships_for_student(student: Student) -> List[Scholarship]:
    """Find all scholarships matching the given student."""
    scholarships = load_scholarships()
    return [s for s in scholarships if s.matches(student)]


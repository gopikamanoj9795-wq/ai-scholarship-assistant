from dataclasses import dataclass


@dataclass
class Student:
    roll_no: str
    category: str
    cgpa: float
    family_income: float
    year: int
    name: str = ""

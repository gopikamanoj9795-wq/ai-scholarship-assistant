from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .student import Student


@dataclass
class Scholarship:
    id: str
    name: str
    category: str
    min_cgpa: float
    max_family_income: float
    eligible_years: List[int] = field(default_factory=list)
    amount: float = 0.0
    deadline: str = ""
    required_docs: List[str] = field(default_factory=list)

    def matches(self, student: Student) -> bool:
        """
        Determines whether a given student meets the eligibility criteria for this scholarship.
        """
        if student.cgpa < self.min_cgpa:
            return False

        if student.family_income > self.max_family_income:
            return False

        if self.eligible_years and student.year not in self.eligible_years:
            return False

        if self.category and self.category.strip().lower() != "all":
            if self.category.strip().lower() != student.category.strip().lower():
                return False

        return True

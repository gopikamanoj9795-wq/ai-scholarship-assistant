from dataclasses import dataclass


@dataclass
class Application:
    roll_no: str
    scholarship_id: str
    status: str = "PENDING"

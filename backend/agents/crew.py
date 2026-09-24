import os
import sys
from pathlib import Path
from datetime import datetime, date
from dataclasses import asdict
from typing import Dict, Any, List

# Disable CrewAI and OpenTelemetry background analytics for local performance
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_KEY"] = "NA"

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool


from backend.config import OLLAMA_HOST, OLLAMA_CREW_MODEL
from backend.data_loader import (
    load_scholarships,
    get_student_by_roll,
    match_scholarships_for_student,
    get_applications_by_roll,
)

# Initialize Ollama LLM for CrewAI with smaller, faster model and timeout safeguard
crew_llm = LLM(
    model=f"ollama/{OLLAMA_CREW_MODEL}",
    base_url=OLLAMA_HOST,
    temperature=0.2,
    timeout=30,
)


@tool("Check Scholarship Eligibility Tool")
def check_eligibility_tool(roll_no: str) -> str:
    """Matches a student's profile against all scholarships and explains eligibility status."""
    student = get_student_by_roll(roll_no)
    if not student:
        return f"Student with roll number '{roll_no}' was not found."

    matched = match_scholarships_for_student(student)
    all_sch = load_scholarships()
    report = [f"Student: {student.name} ({student.roll_no}) | CGPA: {student.cgpa} | Category: {student.category} | Year: {student.year} | Income: {student.family_income}"]
    report.append("\nScholarship Eligibility Breakdown:")

    for s in all_sch:
        is_match = s.matches(student)
        status_str = "ELIGIBLE" if is_match else "INELIGIBLE"
        report.append(f"- {s.name} ({s.id}): {status_str} [Amount: INR {s.amount}, Min CGPA: {s.min_cgpa}, Max Income: {s.max_family_income}, Category: {s.category}, Eligible Years: {s.eligible_years}]")

    return "\n".join(report)


@tool("Track Application Status Tool")
def get_application_status_tool(roll_no: str) -> str:
    """Checks existing submitted applications and returns current review statuses and notes."""
    apps = get_applications_by_roll(roll_no)
    if not apps:
        return f"No existing submitted applications on record for student with roll number '{roll_no}'."

    res = [f"Active Applications for {roll_no}:"]
    for a in apps:
        res.append(f"- Application ID {a.get('application_id')} for Scholarship {a.get('scholarship_id')}: Status '{a.get('status')}', Applied: {a.get('applied_date')}. Notes: {a.get('notes')}")
    return "\n".join(res)


@tool("Check Upcoming Deadlines Tool")
def check_upcoming_deadlines_tool(roll_no: str) -> str:
    """Reviews deadlines for matched scholarships and indicates whether they are urgent (<= 5 days) or on schedule (> 5 days)."""
    student = get_student_by_roll(roll_no)
    if not student:
        return f"Student '{roll_no}' not found."

    matched = match_scholarships_for_student(student)
    today = date.today()
    alerts = []

    for s in matched:
        if s.deadline:
            try:
                deadline_dt = datetime.strptime(s.deadline, "%Y-%m-%d").date()
                days_left = (deadline_dt - today).days
                if 0 <= days_left <= 5:
                    alerts.append(f"URGENT (<= 5 days remaining): '{s.name}' ({s.id}) deadline is in {days_left} day(s) on {s.deadline}.")
                else:
                    alerts.append(f"ON SCHEDULE (> 5 days remaining): '{s.name}' ({s.id}) deadline is in {days_left} day(s) on {s.deadline} (not urgent).")
            except ValueError:
                pass

    if not alerts:
        return "No matched scholarships or upcoming deadlines found."
    return "\n".join(alerts)


# 1. Eligibility Agent
eligibility_agent = Agent(
    role="Scholarship Eligibility Specialist",
    goal="Evaluate student profile against scholarship criteria, determine eligible matches, and explain reasons clearly.",
    backstory="You are an expert academic financial aid counselor with deep knowledge of eligibility rules, income thresholds, and merit criteria.",
    tools=[check_eligibility_tool],
    llm=crew_llm,
    max_iter=3,
    max_execution_time=60,
    max_retry_limit=2,
    verbose=False,
)

# 2. Application Tracker Agent
application_tracker_agent = Agent(
    role="Scholarship Application Tracker",
    goal="Check existing application submissions for the student, report review statuses, and highlight any pending documentation.",
    backstory="You are a meticulous registrar officer tracking university scholarship applications, verification statuses, and disbursement updates.",
    tools=[get_application_status_tool],
    llm=crew_llm,
    max_iter=3,
    max_execution_time=60,
    max_retry_limit=2,
    verbose=False,
)

# 3. Deadline Alert Agent
deadline_alert_agent = Agent(
    role="Scholarship Deadline & Urgency Monitor",
    goal="Review scholarship deadlines strictly. Flag a scholarship as urgent ONLY if days_remaining <= 5. If days_remaining > 5, explicitly report it as 'on schedule' and NOT urgent.",
    backstory="You are a precise student aid advisor who adheres strictly to deadline thresholds: only deadlines with 5 or fewer days remaining (days_remaining <= 5) are urgent, while any deadline more than 5 days away is on schedule.",
    tools=[check_upcoming_deadlines_tool],
    llm=crew_llm,
    max_iter=3,
    max_execution_time=60,
    max_retry_limit=2,
    verbose=False,
)


def create_scholarship_crew() -> Crew:
    """Creates a sequential CrewAI crew with the three scholarship agents."""
    eligibility_task = Task(
        description="Check eligibility for student '{roll_no}'. Use check_eligibility_tool to identify all matching scholarships and explain why the student qualifies.",
        expected_output="A concise explanation of all matching scholarships and eligibility reasons for student '{roll_no}'.",
        agent=eligibility_agent,
    )

    tracking_task = Task(
        description="Check existing submitted application records for student '{roll_no}' using get_application_status_tool and report their current statuses and notes.",
        expected_output="A status report of all current applications for student '{roll_no}'.",
        agent=application_tracker_agent,
    )

    deadline_task = Task(
        description=(
            "Evaluate deadlines for student '{roll_no}' using check_upcoming_deadlines_tool. "
            "CRITICAL RULE: You must ONLY call a deadline 'urgent' if days_remaining <= 5. "
            "If days_remaining > 5, you must state that the deadline is 'on schedule' and not urgent. "
            "Do NOT mark deadlines that are more than 5 days away as urgent under any circumstances."
        ),
        expected_output="A concise deadline summary clearly stating which scholarships are urgent (days_remaining <= 5) versus on schedule (days_remaining > 5).",
        agent=deadline_alert_agent,
    )

    return Crew(
        agents=[eligibility_agent, application_tracker_agent, deadline_alert_agent],
        tasks=[eligibility_task, tracking_task, deadline_task],
        process=Process.sequential,
        verbose=False,
    )


def run_scholarship_crew(roll_no: str) -> Dict[str, Any]:
    """
    Runs the multi-agent scholarship crew for a given student roll number
    and returns structured results alongside combined agent output.
    """
    clean_roll = roll_no.strip()
    student = get_student_by_roll(clean_roll)

    if not student:
        return {
            "roll_no": clean_roll,
            "student": None,
            "matched_scholarships": [],
            "applications": [],
            "urgent_deadlines": [],
            "crew_output": f"Student with roll number '{clean_roll}' was not found.",
        }

    matched = match_scholarships_for_student(student)
    apps = get_applications_by_roll(clean_roll)

    today = date.today()
    urgent: List[Dict[str, Any]] = []
    for s in matched:
        if s.deadline:
            try:
                deadline_dt = datetime.strptime(s.deadline, "%Y-%m-%d").date()
                days_left = (deadline_dt - today).days
                if 0 <= days_left <= 5:
                    urgent_item = asdict(s)
                    urgent_item["days_remaining"] = days_left
                    urgent.append(urgent_item)
            except ValueError:
                pass

    try:
        crew = create_scholarship_crew()
        crew_result = crew.kickoff(inputs={"roll_no": clean_roll})
        crew_output_text = str(crew_result).strip()
    except Exception as e:
        # Fallback in case of timeout or connection error during crew execution
        matched_names = ", ".join([s.name for s in matched]) if matched else "None"
        app_summary = f"{len(apps)} active application(s)" if apps else "No active applications"
        crew_output_text = (
            f"[Agent Synthesis Fallback] Student {student.name} matched with: {matched_names}. "
            f"Application status: {app_summary}. (Agent crew notice: {str(e)})"
        )

    return {
        "roll_no": clean_roll,
        "student": asdict(student),
        "matched_scholarships": [asdict(s) for s in matched],
        "applications": apps,
        "urgent_deadlines": urgent,
        "crew_output": crew_output_text,
    }

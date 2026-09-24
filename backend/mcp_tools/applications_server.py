import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mcp.server.fastmcp import FastMCP
from backend.data_loader import load_applications, get_applications_by_roll

# Initialize FastMCP Server
mcp = FastMCP("Scholarship Applications Server")


@mcp.tool()
def get_application_status(roll_no: str) -> str:
    """
    Retrieves the scholarship application status, submission dates, and notes
    for a student by their roll number.
    
    Args:
        roll_no: The student roll number (e.g. STU001, STU002)
    """
    clean_roll = roll_no.strip().upper()
    apps = get_applications_by_roll(clean_roll)

    if not apps:
        return f"No scholarship applications found on record for student roll number '{clean_roll}'."

    lines = [f"Application records for Student {clean_roll} ({len(apps)} found):"]
    for app in apps:
        app_id = app.get("application_id", "N/A")
        sch_id = app.get("scholarship_id", "N/A")
        status = app.get("status", "Unknown")
        applied_date = app.get("applied_date", "N/A")
        notes = app.get("notes", "No notes available.")
        lines.append(
            f"- Application [{app_id}] for Scholarship [{sch_id}]: "
            f"Status = {status} | Applied = {applied_date} | Notes: {notes}"
        )

    return "\n".join(lines)


@mcp.tool()
def list_pending_applications() -> str:
    """
    Lists all scholarship applications that currently have a status of 'Under Review'
    awaiting administrative or faculty evaluation.
    """
    all_apps = load_applications()
    pending = [
        app for app in all_apps
        if app.get("status", "").strip().lower() == "under review"
    ]

    if not pending:
        return "No pending applications currently with status 'Under Review'."

    lines = [f"Found {len(pending)} pending application(s) awaiting evaluation:"]
    for app in pending:
        app_id = app.get("application_id", "N/A")
        roll_no = app.get("roll_no", "N/A")
        sch_id = app.get("scholarship_id", "N/A")
        status = app.get("status", "Under Review")
        applied_date = app.get("applied_date", "N/A")
        notes = app.get("notes", "No notes available.")
        lines.append(
            f"- Application [{app_id}] | Student: {roll_no} | Scholarship: {sch_id} | "
            f"Status: {status} | Applied: {applied_date} | Notes: {notes}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.mcp_tools.applications_server import (
    get_application_status,
    list_pending_applications,
)


def test_mcp_tools():
    print("--- Testing get_application_status for STU001 ---")
    status_stu1 = get_application_status("STU001")
    print(status_stu1)
    assert "APP101" in status_stu1
    assert "SCH003" in status_stu1
    assert "Under Review" in status_stu1
    print("[PASS] get_application_status('STU001')")

    print("\n--- Testing get_application_status for STU002 ---")
    status_stu2 = get_application_status("STU002")
    print(status_stu2)
    assert "APP102" in status_stu2
    assert "APP103" in status_stu2
    assert "Approved" in status_stu2
    assert "Incomplete" in status_stu2
    print("[PASS] get_application_status('STU002')")

    print("\n--- Testing get_application_status for STU003 (No applications) ---")
    status_stu3 = get_application_status("STU003")
    print(status_stu3)
    assert "No scholarship applications found" in status_stu3
    print("[PASS] get_application_status('STU003')")

    print("\n--- Testing list_pending_applications ---")
    pending = list_pending_applications()
    print(pending)
    assert "APP101" in pending
    assert "Under Review" in pending
    assert "STU001" in pending
    print("[PASS] list_pending_applications()")

    print("\nALL MCP APPLICATION TOOL TESTS PASSED!")


if __name__ == "__main__":
    test_mcp_tools()

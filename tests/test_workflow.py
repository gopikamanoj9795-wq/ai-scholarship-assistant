import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.graph.workflow import run_scholarship_workflow
from backend.agents.crew import run_scholarship_crew
from backend.main import app
from fastapi.testclient import TestClient


def test_crew_and_workflow():
    print("--- Testing CrewAI run_scholarship_crew directly with STU001 ---")
    crew_res = run_scholarship_crew("STU001")
    print("Crew Result keys:", list(crew_res.keys()))
    assert crew_res["roll_no"] == "STU001"
    assert crew_res["student"] is not None
    assert len(crew_res["matched_scholarships"]) == 1
    assert len(crew_res["applications"]) == 1
    assert "crew_output" in crew_res
    print(f"[PASS] Crew results: Matched {len(crew_res['matched_scholarships'])}, Apps {len(crew_res['applications'])}")

    print("\n--- Testing LangGraph + CrewAI workflow with STU001 ---")
    res1 = run_scholarship_workflow("STU001")
    print("Workflow Result keys:", list(res1.keys()))
    assert res1["roll_no"] == "STU001"
    assert res1["student"] is not None
    assert res1["student"]["name"] == "Aarav Sharma"
    assert res1["needs_human_review"] is False
    assert len(res1["matched_scholarships"]) == 1
    assert len(res1["applications"]) == 1
    assert "urgent_deadlines" in res1
    assert "final_response" in res1
    print("[PASS] Direct workflow STU001")

    print("\n--- Testing workflow with unknown student UNKNOWN ---")
    res_unknown = run_scholarship_workflow("UNKNOWN")
    assert res_unknown["roll_no"] == "UNKNOWN"
    assert res_unknown["student"] is None
    assert res_unknown["needs_human_review"] is True
    assert "manual human review" in res_unknown["final_response"].lower()
    print("[PASS] Direct workflow UNKNOWN (Human review branch)")

    print("\n--- Testing FastAPI /crew-workflow/{roll_no} endpoint ---")
    client = TestClient(app)
    res_api = client.get("/crew-workflow/STU001")
    assert res_api.status_code == 200
    data = res_api.json()
    assert data["roll_no"] == "STU001"
    assert data["student"]["name"] == "Aarav Sharma"
    assert data["needs_human_review"] is False
    assert len(data["matched_scholarships"]) == 1
    assert len(data["applications"]) == 1
    print("[PASS] FastAPI endpoint GET /crew-workflow/STU001")

    res_api_unknown = client.get("/crew-workflow/UNKNOWN")
    assert res_api_unknown.status_code == 200
    data_unknown = res_api_unknown.json()
    assert data_unknown["roll_no"] == "UNKNOWN"
    assert data_unknown["student"] is None
    assert data_unknown["needs_human_review"] is True
    print("[PASS] FastAPI endpoint GET /crew-workflow/UNKNOWN")

    print("\nALL CREW & WORKFLOW TESTS PASSED!")


if __name__ == "__main__":
    test_crew_and_workflow()

import sys
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api():
    print("Testing GET /health ...")
    res = client.get("/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert res.json() == {"status": "ok"}, f"Expected {{'status': 'ok'}}, got {res.json()}"
    print("[PASS] GET /health")

    print("\nTesting GET /students ...")
    res = client.get("/students")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    students = res.json()
    assert isinstance(students, list)
    assert len(students) == 3, f"Expected 3 students, got {len(students)}"
    roll_numbers = [s["roll_no"] for s in students]
    assert "STU001" in roll_numbers and "STU002" in roll_numbers and "STU003" in roll_numbers
    print(f"[PASS] GET /students (Found {len(students)} students: {roll_numbers})")

    print("\nTesting GET /scholarships ...")
    res = client.get("/scholarships")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    scholarships = res.json()
    assert isinstance(scholarships, list)
    assert len(scholarships) == 3, f"Expected 3 scholarships, got {len(scholarships)}"
    sch_ids = [s["id"] for s in scholarships]
    assert "SCH001" in sch_ids and "SCH002" in sch_ids and "SCH003" in sch_ids
    print(f"[PASS] GET /scholarships (Found {len(scholarships)} scholarships: {sch_ids})")

    print("\nTesting GET /match/STU001 (Aarav Sharma - General, 8.85 CGPA, 450k Income, Year 3) ...")
    res = client.get("/match/STU001")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["student"]["roll_no"] == "STU001"
    assert data["total_matched"] == 1
    assert data["matches"][0]["id"] == "SCH003"
    print(f"[PASS] GET /match/STU001 (Matched {data['total_matched']} scholarships: {[m['id'] for m in data['matches']]})")

    print("\nTesting GET /match/STU002 (Priya Patel - OBC, 7.60 CGPA, 180k Income, Year 2) ...")
    res = client.get("/match/STU002")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["student"]["roll_no"] == "STU002"
    assert data["total_matched"] == 2
    matched_ids = [m["id"] for m in data["matches"]]
    assert "SCH001" in matched_ids and "SCH002" in matched_ids
    print(f"[PASS] GET /match/STU002 (Matched {data['total_matched']} scholarships: {matched_ids})")

    print("\nTesting GET /match/STU003 (Rahul Kumar - SC, 9.20 CGPA, 220k Income, Year 1) ...")
    res = client.get("/match/STU003")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["student"]["roll_no"] == "STU003"
    assert data["total_matched"] == 1
    assert data["matches"][0]["id"] == "SCH001"
    print(f"[PASS] GET /match/STU003 (Matched {data['total_matched']} scholarships: {[m['id'] for m in data['matches']]})")

    print("\nTesting GET /match/NONEXISTENT (404 Error handling) ...")
    res = client.get("/match/NONEXISTENT")
    assert res.status_code == 404, f"Expected 404, got {res.status_code}"
    print(f"[PASS] GET /match/NONEXISTENT (Returned 404: {res.json()['detail']})")

    print("\nTesting GET /summary/SCH001 (Ollama AI Summary) ...")
    res = client.get("/summary/SCH001")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    summary_data = res.json()
    assert summary_data["scholarship_id"] == "SCH001"
    assert "summary" in summary_data and len(summary_data["summary"].strip()) > 0
    print(f"[PASS] GET /summary/SCH001 (Summary: \"{summary_data['summary']}\")")

    print("\nTesting GET /summary/NONEXISTENT (404 Error handling) ...")
    res = client.get("/summary/NONEXISTENT")
    assert res.status_code == 404, f"Expected 404, got {res.status_code}"
    print(f"[PASS] GET /summary/NONEXISTENT (Returned 404: {res.json()['detail']})")

    print("\nTesting GET /policy-question/SCH001 (RAG Policy Question) ...")
    res = client.get("/policy-question/SCH001?question=What is the appeal process if my application is rejected?")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    qa_data = res.json()
    assert qa_data["scholarship_id"] == "SCH001"
    assert "appeal" in qa_data["question"].lower()
    assert "answer" in qa_data and len(qa_data["answer"].strip()) > 0
    print(f"[PASS] GET /policy-question/SCH001 (Answer snippet: \"{qa_data['answer'][:120]}...\")")

    print("\nTesting GET /policy-question/SCH002 (RAG Policy Question) ...")
    res = client.get("/policy-question/SCH002?question=What are the renewal conditions?")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    qa_data2 = res.json()
    assert qa_data2["scholarship_id"] == "SCH002"
    assert "answer" in qa_data2 and len(qa_data2["answer"].strip()) > 0
    print(f"[PASS] GET /policy-question/SCH002 (Answer snippet: \"{qa_data2['answer'][:120]}...\")")

    print("\nTesting GET /policy-question/NONEXISTENT (404 Error handling) ...")
    res = client.get("/policy-question/NONEXISTENT?question=How do I apply?")
    assert res.status_code == 200 or res.status_code == 404
    assert res.status_code == 404, f"Expected 404, got {res.status_code}"
    print(f"[PASS] GET /policy-question/NONEXISTENT (Returned 404: {res.json()['detail']})")

    print("\nTesting GET /workflow/STU001 (LangGraph Workflow) ...")
    res = client.get("/workflow/STU001")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    wf_data = res.json()
    assert wf_data["roll_no"] == "STU001"
    assert wf_data["student"]["name"] == "Aarav Sharma"
    assert wf_data["needs_human_review"] is False
    assert len(wf_data["matched_scholarships"]) == 1
    assert "final_response" in wf_data and len(wf_data["final_response"]) > 0
    print(f"[PASS] GET /workflow/STU001 (Matched: {len(wf_data['matched_scholarships'])}, Human review: {wf_data['needs_human_review']})")

    print("\nTesting GET /workflow/UNKNOWN (LangGraph Human Review Branch) ...")
    res = client.get("/workflow/UNKNOWN")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    wf_unknown = res.json()
    assert wf_unknown["roll_no"] == "UNKNOWN"
    assert wf_unknown["student"] is None
    assert wf_unknown["needs_human_review"] is True
    assert "manual human review" in wf_unknown["final_response"].lower()
    print("\nTesting GET /crew-workflow/STU001 (CrewAI Multi-Agent Workflow) ...")
    res = client.get("/crew-workflow/STU001")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    crew_data = res.json()
    assert crew_data["roll_no"] == "STU001"
    assert crew_data["student"]["name"] == "Aarav Sharma"
    assert len(crew_data["matched_scholarships"]) == 1
    assert len(crew_data["applications"]) == 1
    assert crew_data["applications"][0]["application_id"] == "APP101"
    assert "final_response" in crew_data and len(crew_data["final_response"]) > 0
    print(f"[PASS] GET /crew-workflow/STU001 (Matches: {len(crew_data['matched_scholarships'])}, Apps: {len(crew_data['applications'])})")

    print("\nTesting GET /crew-workflow/UNKNOWN (CrewAI Human Review Branch) ...")
    res = client.get("/crew-workflow/UNKNOWN")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    crew_unknown = res.json()
    assert crew_unknown["roll_no"] == "UNKNOWN"
    assert crew_unknown["student"] is None
    assert crew_unknown["needs_human_review"] is True
    print(f"[PASS] GET /crew-workflow/UNKNOWN (Needs human review: {crew_unknown['needs_human_review']})")

    print("\nALL FASTAPI ENDPOINT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_api()





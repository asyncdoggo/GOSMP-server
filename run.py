from app.app import app
from fastapi.testclient import TestClient

if __name__ == "__main__":
    # test_client = TestClient(app)

    # response = test_client.get("/")
    # assert response.status_code == 200

    # optimized_data = {
    #     "risk_category": "Low risk",
    #     "risk_score": 0.1,
    #     "invest_amount": 1_00_000,
    #     "duration": 1800 # days
    # }

    # response = test_client.post("/optimize", json=optimized_data)

    # print(response.json())

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
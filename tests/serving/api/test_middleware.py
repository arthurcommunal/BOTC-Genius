from data.generate_test_json import USER_KEY

k=5


def test_count_after_request(client):
    client.post(url="/v1/query",
                headers={"X-API-Key": USER_KEY},
                json={"question": "Here is the question!", "top_k": k})
    r = client.get(url="/metrics")
    assert r.json()["latency"]["latency:/v1/query:2xx"]["count"] == 1



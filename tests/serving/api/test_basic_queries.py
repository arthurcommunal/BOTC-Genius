from dataclasses import replace

from data.generate_test_json import USER_KEY


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    
def test_query_without_key(client):
    r = client.post(url="/v1/query",
                    json={"question": "Here is the question!", "top_k": 5})
    assert r.status_code == 401

def test_query_with_key(client):
    k = 5
    r = client.post(url="/v1/query",
                    headers = {"X-API-Key": USER_KEY},
                    json={"question": "Here is the question!", "top_k": k})
    assert r.status_code == 200
    assert len(r.json()["sections"]) == k
    assert not r.json()["cached"] 

    r = client.post(url="/v1/query",
                    headers = {"X-API-Key": USER_KEY},
                    json={"question": "Here is the question!", "top_k": k})
    assert r.status_code == 200
    assert len(r.json()["sections"]) == k
    assert r.json()["cached"] 


def test_metrics(client):
    r = client.get(url="/metrics")
    assert r.status_code == 200
    assert list(r.json().keys()) == ["counters", "gauges", "ratios", "latency", "histograms", "cache"]
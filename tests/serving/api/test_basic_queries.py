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


def test_429(client, fake_clock):
    k = 5
    for i in range(5):
        r = client.post(url="/v1/query",
                         headers={"X-API-Key": USER_KEY},
                         json={"question": f"Here is the {i} question", "top_k": k})
        assert r.status_code == 200

    r6 = client.post(url="/v1/query",
                     headers={"X-API-Key": USER_KEY},
                     json={"question": "Here is the sixth question", "top_k": k})
    assert r6.status_code == 429
    assert int(r6.headers["Retry-After"]) >= 1
    m = client.get(url="/metrics")
    assert m.json()["counters"]["429_errors"] == 1
    fake_clock.advance(1)
    r7 = client.post(url="/v1/query",
                     headers={"X-API-Key": USER_KEY},
                     json={"question": "Here is the seventh question", "top_k": k})
    assert r7.status_code == 200


from data.generate_test_json import ADMIN_KEY, USER_KEY

k = 5
QUESTION = "Who does the Imp kill?"


def ask(client, key, question=QUESTION):
    return client.post(url="/v1/query",
                       headers={"X-API-Key": key},
                       json={"question": question, "top_k": k})


def invalidate(client, key, tenant_id):
    return client.post(url=f"/v1/admin/tenants/{tenant_id}/cache/invalidate",
                       headers={"X-API-Key": key})


def test_invalidate_as_user_is_forbidden(client):
    r = invalidate(client, USER_KEY, "user")
    assert r.status_code == 403


def test_invalidate_unknown_tenant(client):
    r = invalidate(client, ADMIN_KEY, "ghost")
    assert r.status_code == 404


def test_invalidate_known_tenant_without_entries(client):
    r = invalidate(client, ADMIN_KEY, "user")
    assert r.status_code == 200
    assert r.json() == {"tenant_id": "user", "removed": 0}


def test_invalidate_empties_tenant_cache(client):
    assert not ask(client, USER_KEY).json()["cached"]
    assert ask(client, USER_KEY).json()["cached"]

    r = invalidate(client, ADMIN_KEY, "user")
    assert r.status_code == 200
    assert r.json() == {"tenant_id": "user", "removed": 1}

    assert not ask(client, USER_KEY).json()["cached"]


def test_invalidate_leaves_other_tenants(client):
    ask(client, USER_KEY)
    ask(client, ADMIN_KEY)

    invalidate(client, ADMIN_KEY, "user")

    assert ask(client, ADMIN_KEY).json()["cached"]


def test_cache_is_isolated_between_tenants(client):
    r_user = ask(client, USER_KEY)
    r_admin = ask(client, ADMIN_KEY)
    assert not r_user.json()["cached"]
    assert not r_admin.json()["cached"]
    assert r_user.json()["sections"] == r_admin.json()["sections"]

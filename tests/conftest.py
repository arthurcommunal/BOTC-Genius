from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from botc.serving.api.app import create_app
from botc.serving.api.settings import Settings

BOTC_TENANTS_PATH = Path(__file__).parent.resolve().joinpath("data/tenants.json")
BOTC_FILES_PATH = Path(__file__).parent.resolve().joinpath("data/corpus/")


class FakeClock:
    def __init__(self):
        self.time = 98423

    def __call__(self) -> float:
        return self.time

    def advance(self, s: float) -> None:
        if s < 0:
            raise ValueError("Can't add negative time")
        self.time += s

    def rewind(self, s:float) -> None:
        if s < 0:
            raise ValueError("This function needs a positive float as parameter")
        self.time -= s


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings(fake_clock) -> Settings:
    return Settings(tenants_path=BOTC_TENANTS_PATH,
                    files_path=BOTC_FILES_PATH,
                    clock=fake_clock,
                    embedding_dim=32)


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c

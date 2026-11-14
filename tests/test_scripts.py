"""Command-line scripts that can wipe or replace data (database calls are faked)."""

import pytest

from scripts import ingest_docs, setup_db


@pytest.fixture
def fake_setup(monkeypatch):
    done = []
    monkeypatch.setattr(setup_db, "apply_sql", lambda path: done.append(path.name))
    monkeypatch.setattr("db.seed.main", lambda: done.append("seed"))
    return done


def test_setup_refuses_to_wipe_existing_data_without_reset(fake_setup, monkeypatch):
    monkeypatch.setattr(setup_db, "data_exists", lambda: True)
    with pytest.raises(SystemExit, match="--reset"):
        setup_db.main(["--skip-ingest"])
    assert fake_setup == []


def test_setup_reset_rebuilds_and_if_empty_keeps_data(fake_setup, monkeypatch):
    monkeypatch.setattr(setup_db, "data_exists", lambda: True)
    setup_db.main(["--reset", "--skip-ingest"])
    assert fake_setup == ["schema.sql", "seed", "roles.sql"]
    fake_setup.clear()
    setup_db.main(["--if-empty", "--skip-ingest"])
    assert fake_setup == ["roles.sql"]  # the role rules still reach an existing database


def test_setup_builds_an_empty_database_without_any_flag(fake_setup, monkeypatch):
    monkeypatch.setattr(setup_db, "data_exists", lambda: False)
    setup_db.main(["--skip-ingest"])
    assert fake_setup == ["schema.sql", "seed", "roles.sql"]


@pytest.mark.parametrize(
    "argv, folder, remove_missing",
    [([], None, True), (["other"], "other", False), (["other", "--replace"], "other", True)],
)
def test_ingest_docs_only_replaces_the_index_when_asked(monkeypatch, argv, folder, remove_missing):
    from app.retrieval.ingest import IngestStats

    calls = []
    monkeypatch.setattr(
        ingest_docs,
        "ingest_folder",
        lambda f, remove_missing: calls.append((f, remove_missing)) or IngestStats(),
    )
    ingest_docs.main(argv)
    assert calls == [(folder, remove_missing)]

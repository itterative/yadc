"""Unit tests for DatasetRepository.

Strategy:

- Use the shared :class:`DBConnectionFactory` fixture (with migrations
  applied to a temp-file DB) so the schema is created by the existing
  migrations.
- Build up state by calling the repository's public methods.
- Assert on the repository's public methods (no raw SQL in tests).
"""

from __future__ import annotations

import pytest

from yadc.api.modules.db_connection_factory import DBConnectionFactory
from yadc.api.modules.logging_factory import LoggingFactory
from yadc.api.services.dataset_repository import DatasetRepository


@pytest.fixture
def repo(
    db_connection_factory: DBConnectionFactory,
    logging_factory: LoggingFactory,
) -> DatasetRepository:
    return DatasetRepository(db=db_connection_factory, logging=logging_factory)


# --- Datasets ---


class TestUpsertAndGet:
    def test_upsert_then_get(self, repo):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        result = repo.get_dataset("alpha")
        assert result is not None
        assert result.name == "alpha"
        assert result.source == "import"
        assert result.config_path == "/cfg.toml"

    def test_get_missing_returns_none(self, repo):
        assert repo.get_dataset("nope") is None

    def test_upsert_updates_existing(self, repo):
        repo.upsert_dataset("alpha", "/old.toml", "import")
        repo.upsert_dataset("alpha", "/new.toml", "upload")
        result = repo.get_dataset("alpha")
        assert result is not None
        assert result.source == "upload"
        assert result.config_path == "/new.toml"


class TestListDatasets:
    def test_empty(self, repo):
        assert repo.list_datasets() == []

    def test_orders_by_name(self, repo):
        repo.upsert_dataset("b", "/b.toml", "import")
        repo.upsert_dataset("a", "/a.toml", "import")
        repo.upsert_dataset("c", "/c.toml", "import")
        names = [d.name for d in repo.list_datasets()]
        assert names == ["a", "b", "c"]

    def test_image_count_reflects_stats_update(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        # upsert_image does not bump image_count — that's the role of
        # update_dataset_stats, called at the end of apply_scan_diff.
        assert repo.get_dataset("alpha").image_count == 0
        repo.update_dataset_stats(ds_id, image_count=1)
        assert repo.get_dataset("alpha").image_count == 1


class TestDeleteDataset:
    def test_delete_existing(self, repo):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        assert repo.delete_dataset("alpha") is True
        assert repo.get_dataset("alpha") is None

    def test_delete_missing(self, repo):
        assert repo.delete_dataset("nope") is False

    def test_delete_cascades_to_images(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        assert repo.get_image("alpha", 1) is not None
        repo.delete_dataset("alpha")
        assert repo.get_image("alpha", 1) is None


class TestListStale:
    def test_never_scanned_is_stale(self, repo, db_connection_factory: DBConnectionFactory):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        with db_connection_factory.connection() as conn:
            conn.execute("UPDATE datasets SET last_scanned_t = NULL")
        stale = repo.list_stale(cutoff_t=10_000.0)
        assert [s[1] for s in stale] == ["alpha"]

    def test_recently_scanned_is_not_stale(self, repo):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        # upsert_dataset sets last_scanned_t = unixepoch(), so a cutoff
        # far in the future means nothing is stale.
        assert repo.list_stale(cutoff_t=10_000_000.0) == []

    def test_returns_id_name_config_path(self, repo, db_connection_factory: DBConnectionFactory):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        # Force the row to be stale by setting last_scanned_t to a value
        # far in the past.
        with db_connection_factory.connection() as conn:
            conn.execute("UPDATE datasets SET last_scanned_t = 0.0 WHERE id = ?", (ds_id,))
        stale = repo.list_stale(cutoff_t=1.0)
        assert len(stale) == 1
        assert stale[0] == (ds_id, "alpha", "/cfg.toml")


class TestListAllForWatcher:
    def test_returns_all(self, repo):
        repo.upsert_dataset("a", "/a.toml", "import")
        repo.upsert_dataset("b", None, "import")
        result = repo.list_all_for_watcher()
        assert sorted(result) == [("a", "/a.toml"), ("b", None)]


class TestGetDatasetRow:
    def test_returns_id_config(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        assert repo.get_dataset_row("alpha") == (ds_id, "/cfg.toml")

    def test_missing(self, repo):
        assert repo.get_dataset_row("nope") is None


# --- Images ---


class TestUpsertImage:
    def test_insert(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=True,
            has_toml=False,
            width=100,
            height=200,
            draft_names="alpha",
            last_modified_t=1234.0,
        )
        img = repo.get_image("alpha", 1)
        assert img is not None
        assert img.file_name == "a.jpg"
        assert img.has_caption is True
        assert img.has_toml is False
        assert img.width == 100
        assert img.height == 200
        assert img.draft_names == ["alpha"]
        assert img.last_modified_t == 1234.0

    def test_upsert_updates_existing(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=True,
            has_toml=True,
            width=10,
            height=20,
            draft_names="x",
            last_modified_t=99.0,
        )
        img = repo.get_image("alpha", 1)
        assert img is not None
        assert img.has_caption is True
        assert img.has_toml is True
        assert img.width == 10


class TestListImages:
    def test_paginates_desc(self, repo):
        """First page (large ``before_id``) returns the newest images first."""
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        for i in range(5):
            repo.upsert_image(
                dataset_id=ds_id,
                path=f"/img/{i}.jpg",
                file_name=f"{i}.jpg",
                has_caption=False,
                has_toml=False,
                width=0,
                height=0,
                draft_names="",
                last_modified_t=None,
            )
        # IDs 1..5; DESC with a very large ``before_id`` returns 5, 4, 3, ...
        result = repo.list_images("alpha", before_id=10**9, limit=2)
        assert [r.file_name for r in result] == ["4.jpg", "3.jpg"]

    def test_before_id_cursor(self, repo):
        """Cursor ``before_id`` returns images with id < cursor, in DESC order."""
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        for i in range(5):
            repo.upsert_image(
                dataset_id=ds_id,
                path=f"/img/{i}.jpg",
                file_name=f"{i}.jpg",
                has_caption=False,
                has_toml=False,
                width=0,
                height=0,
                draft_names="",
                last_modified_t=None,
            )
        # IDs 1..5; before_id=4 returns id < 4 → ids 1, 2, 3 in DESC order
        result = repo.list_images("alpha", before_id=4, limit=10)
        assert [r.file_name for r in result] == ["2.jpg", "1.jpg", "0.jpg"]

    def test_empty(self, repo):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        assert repo.list_images("alpha", before_id=10**9, limit=10) == []


class TestListImagePathsDesc:
    """``list_image_paths_desc`` — single SQL query used by the
    captioning service to reorder the filesystem-resolved image
    list (newest first) without N+1 callbacks."""

    def test_returns_paths_in_desc_order(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        for i in range(4):
            repo.upsert_image(
                dataset_id=ds_id,
                path=f"/img/{i}.jpg",
                file_name=f"{i}.jpg",
                has_caption=False,
                has_toml=False,
                width=0,
                height=0,
                draft_names="",
                last_modified_t=None,
            )
        result = repo.list_image_paths_desc("alpha")
        # ids 1..4 → paths in DESC id order
        assert result == [("/img/3.jpg", 4), ("/img/2.jpg", 3), ("/img/1.jpg", 2), ("/img/0.jpg", 1)]

    def test_empty(self, repo):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        assert repo.list_image_paths_desc("alpha") == []


class TestGetImageByPath:
    def test_found(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        img = repo.get_image_by_path("alpha", "/img/a.jpg")
        assert img is not None
        assert img.file_name == "a.jpg"

    def test_missing(self, repo):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        assert repo.get_image_by_path("alpha", "/img/nope.jpg") is None


class TestGetImagePath:
    def test_found(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        assert repo.get_image_path("alpha", 1) == "/img/a.jpg"

    def test_missing(self, repo):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        assert repo.get_image_path("alpha", 999) is None


class TestDeleteImage:
    def test_removes_row(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        repo.delete_image(1)
        assert repo.get_image("alpha", 1) is None


class TestUpdateImageFlags:
    def _add(self, repo) -> int:
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        return 1

    def test_caption_only(self, repo):
        image_id = self._add(repo)
        repo.update_image_flags(image_id, has_caption=True)
        img = repo.get_image("alpha", image_id)
        assert img is not None
        assert img.has_caption is True
        assert img.has_toml is False

    def test_toml_only(self, repo):
        image_id = self._add(repo)
        repo.update_image_flags(image_id, has_toml=True)
        img = repo.get_image("alpha", image_id)
        assert img is not None
        assert img.has_caption is False
        assert img.has_toml is True

    def test_both(self, repo):
        image_id = self._add(repo)
        repo.update_image_flags(image_id, has_caption=True, has_toml=True)
        img = repo.get_image("alpha", image_id)
        assert img is not None
        assert img.has_caption is True
        assert img.has_toml is True

    def test_no_op_when_neither_set(self, repo):
        image_id = self._add(repo)
        repo.update_image_flags(image_id)  # neither kwarg set
        img = repo.get_image("alpha", image_id)
        assert img is not None
        # Existing values untouched.
        assert img.has_caption is False
        assert img.has_toml is False

    def test_existing_untouched_when_no_op(self, repo):
        image_id = self._add(repo)
        repo.update_image_flags(image_id, has_caption=True, has_toml=True)
        repo.update_image_flags(image_id)  # neither kwarg set
        img = repo.get_image("alpha", image_id)
        assert img is not None
        assert img.has_caption is True
        assert img.has_toml is True


class TestRefreshImageDiskState:
    def test_updates_fields(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        repo.refresh_image_disk_state(
            1,
            has_caption=True,
            has_toml=False,
            draft_names="alpha,gemma",
        )
        img = repo.get_image("alpha", 1)
        assert img is not None
        assert img.has_caption is True
        assert img.has_toml is False
        assert img.draft_names == ["alpha", "gemma"]


class TestListDraftNamesCsv:
    def test_empty(self, repo):
        repo.upsert_dataset("alpha", "/cfg.toml", "import")
        assert repo.list_draft_names_csv("alpha") == []

    def test_returns_distinct_csvs(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="alpha,gemma",
            last_modified_t=None,
        )
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/b.jpg",
            file_name="b.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="alpha,llama",
            last_modified_t=None,
        )
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/c.jpg",
            file_name="c.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="gemma,llama",
            last_modified_t=None,
        )
        result = set(repo.list_draft_names_csv("alpha"))
        assert result == {"alpha,gemma", "alpha,llama", "gemma,llama"}

    def test_excludes_empty_csvs(self, repo):
        ds_id = repo.upsert_dataset("alpha", "/cfg.toml", "import")
        repo.upsert_image(
            dataset_id=ds_id,
            path="/img/a.jpg",
            file_name="a.jpg",
            has_caption=False,
            has_toml=False,
            width=0,
            height=0,
            draft_names="",
            last_modified_t=None,
        )
        assert repo.list_draft_names_csv("alpha") == []

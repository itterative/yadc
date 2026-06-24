"""Tests for the SSE events dispatched by ``TaggingService``.

The ``EventDispatcher`` is mocked so we can inspect dispatched events
without booting the full Quart app.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from yadc.api.events import ImageTagErrorEvent, ImageTaggedEvent, TaggerStatusEvent
from yadc.api.modules import EventDispatcher, LoggingFactory
from yadc.api.services.dataset_jobs import DatasetJobService
from yadc.api.services.dataset_repository import ImageInfo
from yadc.api.services.tagging import TaggingService
from yadc.taggers.base import TaggerResult

from .conftest import make_client_mock, patch_client_factory

# ---------------------------------------------------------------------------
# Fixtures local to this file (image fixtures and the subprocess mock /
# patcher come from the directory's conftest.py).
# ---------------------------------------------------------------------------


@pytest.fixture
def event_dispatcher() -> tuple[EventDispatcher, MagicMock]:
    """A real EventDispatcher paired with a mock ``dispatch`` spy.

    Returns ``(dispatcher, dispatch_spy)``. The TaggingService should
    be constructed with the dispatcher; tests inspect ``dispatch_spy.call_args_list``.
    """
    logging_factory = MagicMock(spec=LoggingFactory)
    logging_factory.get_logger = MagicMock(return_value=MagicMock())
    dispatcher = EventDispatcher(logging_factory)
    spy = MagicMock()
    dispatcher.dispatch = spy  # type: ignore[method-assign]
    return dispatcher, spy


@pytest.fixture
def service(
    test_configuration,
    logging_factory: LoggingFactory,
    event_dispatcher: tuple[EventDispatcher, MagicMock],
    dataset_service: MagicMock,
    dataset_watcher: MagicMock,
    dataset_jobs: DatasetJobService,
) -> TaggingService:
    test_configuration.tagger_model_path = "/fake/model.onnx"
    test_configuration.tagger_repo_id = ""
    test_configuration.tagger_idle_timeout_seconds = 60.0
    return TaggingService(
        logging=logging_factory,
        event_dispatcher=event_dispatcher[0],
        configuration=test_configuration,
        dataset_service=dataset_service,
        dataset_watcher=dataset_watcher,
        job_scheduler=None,
        dataset_jobs=dataset_jobs,
    )


def _dispatched(spy: MagicMock) -> list[Any]:
    """Return the list of events ``dispatch`` was called with."""
    return [c.args[0] for c in spy.call_args_list]


# ---------------------------------------------------------------------------
# TaggerStatusEvent lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_first_request_dispatches_starting_then_ready(
        self,
        service: TaggingService,
        image_info: ImageInfo,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
    ) -> None:
        """First request: subprocess spawns → starting → ready → tag fires."""
        client = make_client_mock(alive=True)
        _, dispatch_spy = event_dispatcher
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))

        statuses = [e for e in _dispatched(dispatch_spy) if isinstance(e, TaggerStatusEvent)]
        assert [e.state for e in statuses] == ["starting", "ready"]

    def test_subsequent_request_does_not_re_emit_starting(
        self,
        service: TaggingService,
        make_image_info,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
    ) -> None:
        """Reusing the alive subprocess does not emit ``starting`` / ``ready`` again."""
        client = make_client_mock(alive=True)
        _, dispatch_spy = event_dispatcher
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", make_image_info(image_id=42)))
            dispatch_spy.reset_mock()
            asyncio.run(service.tag_image("ds", make_image_info(image_id=43)))

        statuses = [e for e in _dispatched(dispatch_spy) if isinstance(e, TaggerStatusEvent)]
        assert statuses == []

    def test_respawn_after_subprocess_dies_re_dispatches_starting(
        self,
        service: TaggingService,
        make_image_info,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
    ) -> None:
        """When the subprocess dies between requests, the next request re-emits ``starting`` / ``ready``."""
        _, dispatch_spy = event_dispatcher
        client1 = make_client_mock(alive=True)
        with patch_client_factory(client1):
            asyncio.run(service.tag_image("ds", make_image_info(image_id=42)))

        client1.is_alive = False
        client2 = make_client_mock(alive=True)
        with patch_client_factory(client2):
            asyncio.run(service.tag_image("ds", make_image_info(image_id=43)))

        statuses = [e for e in _dispatched(dispatch_spy) if isinstance(e, TaggerStatusEvent)]
        assert [e.state for e in statuses] == ["starting", "ready", "starting", "ready"]

    def test_start_failure_dispatches_failed(
        self,
        service: TaggingService,
        image_info: ImageInfo,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
    ) -> None:
        """If the subprocess fails to start, ``TaggerStatusEvent(state=failed)`` is emitted."""
        client = make_client_mock(alive=True)
        client.start.side_effect = RuntimeError("boom")
        _, dispatch_spy = event_dispatcher

        with patch_client_factory(client):
            with pytest.raises(RuntimeError, match="Failed to start tagger process"):
                asyncio.run(service.tag_image("ds", image_info))

        statuses = [e for e in _dispatched(dispatch_spy) if isinstance(e, TaggerStatusEvent)]
        assert [e.state for e in statuses] == ["starting", "failed"]
        assert statuses[-1].error == "boom"

    def test_shutdown_dispatches_stopping_then_stopped(
        self,
        service: TaggingService,
        image_info: ImageInfo,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
    ) -> None:
        """App shutdown tears down the subprocess and dispatches the corresponding events."""
        client = make_client_mock(alive=True)
        _, dispatch_spy = event_dispatcher
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
            dispatch_spy.reset_mock()
            asyncio.run(service.on_shutdown(None))

        statuses = [e for e in _dispatched(dispatch_spy) if isinstance(e, TaggerStatusEvent)]
        assert [e.state for e in statuses] == ["stopping", "stopped"]

    def test_idle_teardown_dispatches_stopping_then_stopped(
        self,
        service: TaggingService,
        image_info: ImageInfo,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Idle-check teardown also dispatches ``stopping`` / ``stopped``."""
        client = make_client_mock(alive=True)
        _, dispatch_spy = event_dispatcher
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))
            dispatch_spy.reset_mock()

            last_used = service._last_used_t
            assert last_used is not None
            monkeypatch.setattr(time, "monotonic", lambda: last_used + service._configuration.tagger_idle_timeout_seconds + 1)
            service._idle_check_tick()

        statuses = [e for e in _dispatched(dispatch_spy) if isinstance(e, TaggerStatusEvent)]
        assert [e.state for e in statuses] == ["stopping", "stopped"]


# ---------------------------------------------------------------------------
# ImageTaggedEvent / ImageTagErrorEvent
# ---------------------------------------------------------------------------


class TestEvents:
    def test_successful_tag_dispatches_image_tagged_event(
        self,
        service: TaggingService,
        image_info: ImageInfo,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
    ) -> None:
        """A successful tag emits an ``ImageTaggedEvent`` with the thresholded result and frontend-supplied source."""
        client = make_client_mock(
            alive=True,
            tag_result=TaggerResult(
                tags={"1girl": 0.9, "smile": 0.1, "safe": 0.99},
                categories={"general": ["1girl", "smile"], "rating": ["safe"]},
            ),
        )
        _, dispatch_spy = event_dispatcher
        with patch_client_factory(client):
            asyncio.run(
                service.tag_image(
                    "ds",
                    image_info,
                    source="ui:wd-swinv2-tagger-v3",
                )
            )

        tagged = [e for e in _dispatched(dispatch_spy) if isinstance(e, ImageTaggedEvent)]
        assert len(tagged) == 1
        event = tagged[0]
        assert event.dataset_name == "ds"
        assert event.image_id == image_info.id
        assert event.file_name == image_info.file_name
        assert event.path == image_info.path
        # General threshold 0.35 drops "smile" (0.1).
        assert "smile" not in event.tags
        assert "1girl" in event.tags
        assert event.categories["general"] == ["1girl"]
        assert event.source == "ui:wd-swinv2-tagger-v3"
        assert event.duration_ms >= 0

    def test_tag_event_source_falls_back_to_configured_model_when_not_supplied(
        self,
        service: TaggingService,
        image_info: ImageInfo,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
    ) -> None:
        """When the caller doesn't supply ``source``, the event falls back to the server-configured model."""
        client = make_client_mock(alive=True)
        _, dispatch_spy = event_dispatcher
        with patch_client_factory(client):
            asyncio.run(service.tag_image("ds", image_info))

        tagged = [e for e in _dispatched(dispatch_spy) if isinstance(e, ImageTaggedEvent)]
        assert tagged[0].source == "local:/fake/model.onnx"

    def test_tag_failure_dispatches_image_tag_error_event(
        self,
        service: TaggingService,
        image_info: ImageInfo,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
    ) -> None:
        """A tag that raises during inference emits ``ImageTagErrorEvent``."""
        client = make_client_mock(alive=True)
        client.tag.side_effect = RuntimeError("inference crashed")
        _, dispatch_spy = event_dispatcher

        with patch_client_factory(client):
            with pytest.raises(RuntimeError, match="inference crashed"):
                asyncio.run(
                    service.tag_image(
                        "ds",
                        image_info,
                        source="ui:wd-vit-tagger-v2",
                    )
                )

        errors = [e for e in _dispatched(dispatch_spy) if isinstance(e, ImageTagErrorEvent)]
        assert len(errors) == 1
        assert errors[0].error == "inference crashed"
        assert errors[0].image_id == image_info.id
        assert errors[0].source == "ui:wd-vit-tagger-v2"
        assert errors[0].duration_ms >= 0

    def test_unconfigured_does_not_dispatch_event(
        self,
        test_configuration,
        logging_factory: LoggingFactory,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
        dataset_service: MagicMock,
        dataset_watcher: MagicMock,
        dataset_jobs: DatasetJobService,
        image_info: ImageInfo,
    ) -> None:
        """A request with no model configured raises ``RuntimeError`` without dispatching."""
        _, dispatch_spy = event_dispatcher
        test_configuration.tagger_model_path = ""
        test_configuration.tagger_repo_id = ""
        s = TaggingService(
            logging=logging_factory,
            event_dispatcher=event_dispatcher[0],
            configuration=test_configuration,
            dataset_service=dataset_service,
            dataset_watcher=dataset_watcher,
            job_scheduler=None,
            dataset_jobs=dataset_jobs,
        )

        with pytest.raises(RuntimeError, match="not configured"):
            asyncio.run(s.tag_image("ds", image_info))

        assert _dispatched(dispatch_spy) == []


# ---------------------------------------------------------------------------
# Source label
# ---------------------------------------------------------------------------


class TestSourceLabel:
    def test_source_label_for_local_model(self, service: TaggingService) -> None:
        assert service._source_label() == "local:/fake/model.onnx"

    def test_source_label_for_hf_repo(
        self,
        test_configuration,
        logging_factory: LoggingFactory,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
        dataset_service: MagicMock,
        dataset_watcher: MagicMock,
        dataset_jobs: DatasetJobService,
    ) -> None:
        test_configuration.tagger_model_path = ""
        test_configuration.tagger_repo_id = "SmilingWolf/wd-v1-4-vit-tagger-v2"
        s = TaggingService(
            logging=logging_factory,
            event_dispatcher=event_dispatcher[0],
            configuration=test_configuration,
            dataset_service=dataset_service,
            dataset_watcher=dataset_watcher,
            job_scheduler=None,
            dataset_jobs=dataset_jobs,
        )
        assert s._source_label() == "hf:SmilingWolf/wd-v1-4-vit-tagger-v2"

    def test_source_label_when_unconfigured(
        self,
        test_configuration,
        logging_factory: LoggingFactory,
        event_dispatcher: tuple[EventDispatcher, MagicMock],
        dataset_service: MagicMock,
        dataset_watcher: MagicMock,
        dataset_jobs: DatasetJobService,
    ) -> None:
        test_configuration.tagger_model_path = ""
        test_configuration.tagger_repo_id = ""
        s = TaggingService(
            logging=logging_factory,
            event_dispatcher=event_dispatcher[0],
            configuration=test_configuration,
            dataset_service=dataset_service,
            dataset_watcher=dataset_watcher,
            job_scheduler=None,
            dataset_jobs=dataset_jobs,
        )
        assert s._source_label() == ""

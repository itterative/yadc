"""Tests for the tagger subprocess server."""

import pytest

from yadc.taggers.base import Tagger, TaggerResult
from yadc.taggers.server import TaggerServer


class DummyTagger(Tagger):
    """A no-op tagger for testing. Returns the same fixed result every time."""

    def __init__(self) -> None:
        self.loaded = False

    def load_model(self, model_path: str, **kwargs) -> None:  # noqa: ANN001, ANN401
        self.loaded = True

    def unload_model(self) -> None:
        self.loaded = False

    def predict(self, image_bytes: bytes) -> TaggerResult:
        return TaggerResult(
            tags={"tag1": 0.9, "tag2": 0.1, "rating:safe": 0.99},
            categories={"rating": ["rating:safe"]},
        )


class TestServer:
    @pytest.fixture
    def server(self):
        return TaggerServer(DummyTagger, "/fake/model.onnx")

    def test_tagger_server_start_stop(self, server) -> None:
        """Test that the server starts, tags, and stops cleanly."""
        server.start()
        try:
            assert server.is_alive
            result = server.tag(b"fake_image_bytes")
            assert isinstance(result, TaggerResult)
            assert result.tags["tag1"] == 0.9
            assert result.categories == {"rating": ["rating:safe"]}
        finally:
            server.stop()
        assert not server.is_alive

    def test_tagger_server_stop_without_start(self, server) -> None:
        """Test that stop() is safe when the server was never started."""
        server.stop()  # should not raise

    def test_tagger_server_kill_without_start(self, server) -> None:
        """``kill()`` is safe when the server was never started."""
        server.kill()  # should not raise

    def test_tagger_server_tag_before_start(self, server) -> None:
        """Test that tag() raises before start()."""
        try:
            server.start()
            # Save original queue so stop() can send sentinel.
            orig_rq = server._request_queue
            server._request_queue = None
            server._response_queue = None
            with pytest.raises(RuntimeError, match="not started"):
                server.tag(b"image")
        finally:
            server._request_queue = orig_rq  # restore so stop() works
            server.stop()

    def test_tagger_server_tag_drains_heartbeat(self, server) -> None:
        """Control messages on the response queue are skipped, not mistaken for a result."""
        server.start()
        try:
            # Inject a heartbeat before the real response lands. The drain
            # loop must skip the ``id == 0`` control message and return the
            # matching result for our request id.
            server._response_queue.put({"id": 0, "heartbeat": True})
            result = server.tag(b"fake_image_bytes")
            assert result.tags["tag1"] == 0.9
        finally:
            server.stop()


class FailingTagger(Tagger):
    """A tagger whose ``load_model`` always raises — for testing failure paths."""

    def load_model(self, model_path: str, **kwargs) -> None:
        raise RuntimeError("model not found")

    def unload_model(self) -> None:
        pass

    def predict(self, image_bytes: bytes) -> TaggerResult:
        return TaggerResult(tags={})


class TestFailingTagger:
    @pytest.fixture
    def server(self):
        return TaggerServer(FailingTagger, "/missing/model.onnx")

    def test_tagger_server_model_load_failure(self, server) -> None:
        """Test that server handles model load failure."""
        try:
            server.start()
        except RuntimeError as exc:
            assert "model not found" in str(exc)
        assert not server.is_alive


class HangingTagger(Tagger):
    """A tagger whose ``predict`` never returns — for testing worker-death detection."""

    def load_model(self, model_path: str, **kwargs) -> None:  # noqa: ANN001, ANN401
        pass

    def unload_model(self) -> None:
        pass

    def predict(self, image_bytes: bytes) -> TaggerResult:
        import os

        # Kill the worker process mid-request so the main process sees
        # a dead pipe / failed is_alive() instead of a response.
        os.kill(os.getpid(), 9)
        return TaggerResult(tags={})  # unreachable


class TestHangingServer:
    @pytest.fixture
    def server(self):
        return TaggerServer(
            HangingTagger,
            "/fake/model.onnx",
            # Dead-worker detection is gated by ``poll_interval`` (the parent
            # re-checks ``is_alive()`` each poll), not ``heartbeat_interval``;
            # ``HangingTagger.predict`` kills the process immediately, so the
            # idle heartbeat never fires. Cranked low to keep this test fast
            # — the production defaults (15s / 1.0s) are irrelevant here.
            heartbeat_interval=0.05,
            poll_interval=0.05,
            response_timeout=5.0,
        )

    def test_tagger_server_tag_detects_dead_worker(self, server) -> None:
        """``tag()`` raises rather than hanging when the worker dies mid-request."""
        server.start()
        try:
            with pytest.raises(RuntimeError):
                server.tag(b"fake_image_bytes")
        finally:
            server.stop()
        assert not server.is_alive

    def test_tagger_server_kill_terminates_hanging_worker(self, server) -> None:
        """``kill()`` ends a worker that ignores the graceful sentinel (a worker
        mid-native-call won't service ``None`` until it returns). Unlike
        ``stop()``, ``kill()`` skips the sentinel and force-terminates."""
        server.start()
        assert server.is_alive
        # ``kill`` should return promptly (no graceful join) and leave the
        # process dead, even though the worker would otherwise hang.
        server.kill()
        assert not server.is_alive

import asyncio

from app.services import gpu_status


def _fresh(monkeypatch) -> None:
    monkeypatch.setattr(gpu_status, "_status", dict(gpu_status._status))
    monkeypatch.setattr(gpu_status, "_fails", 0)


def test_chunk_results_update_status(monkeypatch) -> None:
    _fresh(monkeypatch)
    gpu_status.mark_gpu(True)
    assert gpu_status.gpu_status()["reachable"] is True
    gpu_status.mark_gpu(False, "tunnel down")
    status = gpu_status.gpu_status()
    assert status["reachable"] is False
    assert status["error"] == "tunnel down"


def test_check_interval_backs_off_after_failures(monkeypatch) -> None:
    # Circuit breaker: 20 s while healthy, then 40 s, 80 s, capped at 120 s, back to 20 s on success.
    _fresh(monkeypatch)
    delays = []
    for _ in range(6):
        gpu_status.mark_gpu(False)
        delays.append(gpu_status.next_check_delay())
    assert delays == [20.0, 20.0, 40.0, 80.0, 120.0, 120.0]
    gpu_status.mark_gpu(True)
    assert gpu_status.next_check_delay() == 20.0


def test_check_gpu_reads_kaggle_health(monkeypatch) -> None:
    _fresh(monkeypatch)
    from app.services import chatterbox_client

    async def down() -> dict:
        return {"reachable": False, "cuda": False, "loaded": False, "error": "502"}

    async def up() -> dict:
        return {"reachable": True, "cuda": True, "loaded": True, "error": None}

    monkeypatch.setattr(chatterbox_client, "ping_colab", down)
    assert asyncio.run(gpu_status.check_gpu())["reachable"] is False
    monkeypatch.setattr(chatterbox_client, "ping_colab", up)
    status = asyncio.run(gpu_status.check_gpu())
    assert status["reachable"] is True and status["loaded"] is True


def test_health_answers_without_calling_kaggle(monkeypatch) -> None:
    _fresh(monkeypatch)
    from app import main
    from app.services import chatterbox_client

    async def must_not_run() -> dict:
        raise AssertionError("/health must not call Kaggle")

    monkeypatch.setattr(chatterbox_client, "ping_colab", must_not_run)
    monkeypatch.setattr(main, "colab_configured", lambda: True)
    gpu_status.mark_gpu(True)
    assert main.health()["colabReachable"] is True
    assert main.ping() == {"ok": True}

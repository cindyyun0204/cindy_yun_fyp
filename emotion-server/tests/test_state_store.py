"""Tests for the _StateStore class in server.py.

These focus on the audit's concerns: overlapping requests must not corrupt
state or cross sessions, and basic state transitions should be sound.
"""
import asyncio
import pytest

from server import _StateStore, IterationSignal


def _signal(iteration=1, user_id="u1", condition_id="1", group_id="RuralUrbanHighway",
            environment_index=0, condition="combination") -> IterationSignal:
    return IterationSignal(
        iteration=iteration,
        user_id=user_id,
        condition_id=condition_id,
        group_id=group_id,
        environment_index=environment_index,
        condition=condition,
    )


@pytest.fixture
def store():
    return _StateStore()


@pytest.mark.asyncio
async def test_initial_state_is_idle(store):
    s = await store.status_dict()
    assert s["state"] == "idle"
    assert s["iteration"] == 0
    assert s["logs_count"] == 0


@pytest.mark.asyncio
async def test_start_then_append_then_process(store):
    await store.start_iteration(_signal(iteration=3, user_id="alice"))
    s = await store.status_dict()
    assert s["state"] == "collecting"
    assert s["iteration"] == 3
    assert s["user_id"] == "alice"

    accepted, n, _ = await store.append_log({"x": 1})
    assert accepted is True
    assert n == 1

    latest, n_logs, duration, iter_n = await store.begin_processing()
    assert latest == {"x": 1}
    assert n_logs == 1
    assert iter_n == 3
    assert duration >= 0

    s = await store.status_dict()
    assert s["state"] == "processing"

    await store.set_result({"emotion": "happy", "confidence": 0.9})
    s = await store.status_dict()
    assert s["state"] == "ready"
    assert (await store.get_latest_emotion())["emotion"] == "happy"


@pytest.mark.asyncio
async def test_append_rejected_when_idle(store):
    accepted, _, _ = await store.append_log({"x": 1})
    assert accepted is False


@pytest.mark.asyncio
async def test_begin_processing_with_no_logs(store):
    await store.start_iteration(_signal())
    latest, n, _, _ = await store.begin_processing()
    assert latest is None
    assert n == 0
    s = await store.status_dict()
    # No data → flips straight to ready, not processing.
    assert s["state"] == "ready"


@pytest.mark.asyncio
async def test_buffer_drops_oldest_when_full(store):
    """Module sets _MAX_LOGS_PER_ITERATION = 256. Push more than that."""
    from server import _MAX_LOGS_PER_ITERATION

    await store.start_iteration(_signal())
    for i in range(_MAX_LOGS_PER_ITERATION + 50):
        await store.append_log({"i": i})

    s = await store.status_dict()
    assert s["logs_count"] == _MAX_LOGS_PER_ITERATION

    latest, n, _, _ = await store.begin_processing()
    # Newest survives; oldest dropped.
    assert latest == {"i": _MAX_LOGS_PER_ITERATION + 49}


@pytest.mark.asyncio
async def test_overlapping_iterations_clear_old_buffer(store):
    """Audit point: 'overlapping sessions could corrupt state'."""
    await store.start_iteration(_signal(iteration=1, user_id="alice"))
    await store.append_log({"from": "alice"})

    # New iteration arrives before the old one finishes.
    await store.start_iteration(_signal(iteration=2, user_id="bob"))
    s = await store.status_dict()
    assert s["iteration"] == 2
    assert s["user_id"] == "bob"
    assert s["logs_count"] == 0  # alice's log discarded

    latest, n, _, _ = await store.begin_processing()
    assert latest is None  # buffer was cleared on new start


@pytest.mark.asyncio
async def test_concurrent_appends_do_not_lose_or_double_writes(store):
    """50 coroutines append concurrently. Total accepted should equal the
    number of concurrent appends (assuming none exceed the buffer cap)."""
    await store.start_iteration(_signal())

    async def worker(i: int):
        await store.append_log({"i": i})

    await asyncio.gather(*(worker(i) for i in range(50)))

    s = await store.status_dict()
    assert s["logs_count"] == 50
    assert s["state"] == "collecting"


@pytest.mark.asyncio
async def test_force_ready(store):
    await store.start_iteration(_signal())
    await store.force_ready()
    s = await store.status_dict()
    assert s["state"] == "ready"

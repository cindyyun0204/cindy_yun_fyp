"""Tests for tightened request validation in server.py."""
import pytest
from pydantic import ValidationError

from server import EmotionLogRequest, IterationSignal


# --- IterationSignal ------------------------------------------------------

def test_iteration_signal_requires_non_empty_user_id():
    with pytest.raises(ValidationError):
        IterationSignal(
            iteration=1,
            user_id="",          # empty rejected
            condition_id="1",
            group_id="RuralUrbanHighway",
        )


def test_iteration_signal_requires_non_empty_condition_id():
    with pytest.raises(ValidationError):
        IterationSignal(
            iteration=1,
            user_id="alice",
            condition_id="",     # empty rejected
            group_id="RuralUrbanHighway",
        )


def test_iteration_signal_strips_whitespace():
    sig = IterationSignal(
        iteration=1,
        user_id="  alice  ",
        condition_id="1",
        group_id="RuralUrbanHighway",
    )
    assert sig.user_id == "alice"


def test_iteration_signal_rejects_unknown_condition():
    with pytest.raises(ValidationError):
        IterationSignal(
            iteration=1, user_id="alice",
            condition_id="1", group_id="RuralUrbanHighway",
            condition="excited",   # not one of the allowed values
        )


def test_iteration_signal_accepts_empty_condition_string():
    sig = IterationSignal(
        iteration=1, user_id="alice",
        condition_id="1", group_id="RuralUrbanHighway",
        # default condition="" is fine — the C# side may not always set it
    )
    assert sig.condition == ""


def test_iteration_signal_rejects_negative_iteration():
    with pytest.raises(ValidationError):
        IterationSignal(
            iteration=-1, user_id="alice",
            condition_id="1", group_id="RuralUrbanHighway",
        )


# --- EmotionLogRequest ----------------------------------------------------

def test_emotion_log_minimal_payload_accepted():
    req = EmotionLogRequest(session_id="s1", t_utc="2025-01-01T00:00:00Z")
    assert req.face_features == {}
    assert req.face_jpeg_b64 == ""


def test_emotion_log_rejects_empty_session_id():
    with pytest.raises(ValidationError):
        EmotionLogRequest(session_id="", t_utc="2025-01-01T00:00:00Z")


def test_emotion_log_rejects_empty_t_utc():
    with pytest.raises(ValidationError):
        EmotionLogRequest(session_id="s1", t_utc="")


def test_emotion_log_caps_blendshape_dict_size():
    too_many = {f"shape_{i}": 0.1 for i in range(100)}
    with pytest.raises(ValidationError):
        EmotionLogRequest(
            session_id="s1", t_utc="2025-01-01T00:00:00Z",
            blendshapes=too_many,
        )


def test_emotion_log_rejects_oversized_face_jpeg():
    big_b64 = "A" * (10_000_001)
    with pytest.raises(ValidationError):
        EmotionLogRequest(
            session_id="s1", t_utc="2025-01-01T00:00:00Z",
            face_jpeg_b64=big_b64,
        )


def test_emotion_log_rejects_long_asr_text():
    with pytest.raises(ValidationError):
        EmotionLogRequest(
            session_id="s1", t_utc="2025-01-01T00:00:00Z",
            asr_text="x" * 5000,
        )

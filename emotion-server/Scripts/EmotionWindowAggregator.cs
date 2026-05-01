using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class EmotionWindowAggregator : MonoBehaviour
{
    [Header("References")]
    public EmotionClient emotionClient;

    [Header("Window")]
    public float windowSec = 30f;
    public float keepSec = 35f;

    [Header("Blink detection")]
    public float eyeClosedThreshold = 0.18f;

    [Header("Send settings")]
    public bool drivingSession = true;
    public bool sendEvenIfTextEmpty = true;
    public bool sendOnAsrChunk = false;

    [Header("Iteration sync")]
    public string emotionServerUrl = "http://127.0.0.1:8000";
    public float statusPollInterval = 1f;
    public float collectSendInterval = 5f;

    [Header("Testing: Initial quick capture")]
    public float initialCaptureSec = 3f;
    public bool doInitialCapture = true;

    private readonly List<Sample> _buf = new List<Sample>(2048);

    private struct Sample
    {
        public float t;
        public FaceFeatures f;
    }

    private bool _requestInFlight = false;
    private bool _initialCaptureDone = false;
    private float _startTime = 0f;

    // Iteration state tracking
    private string _currentState = "idle";
    private string _currentCondition = "";
    private int _currentIteration = 0;
    private float _lastStatusPoll = 0f;
    private float _lastCollectSend = 0f;
    private bool _sentForThisIteration = false;

    void Awake()
    {
        if (emotionClient == null) emotionClient = GetComponent<EmotionClient>();
        if (emotionClient == null) emotionClient = FindObjectOfType<EmotionClient>();
    
        var envUrl = Environment.GetEnvironmentVariable("EMOTION_SERVER_BASE_URL");
        if (!string.IsNullOrEmpty(envUrl)) emotionServerUrl = envUrl.TrimEnd('/');
    }

    void OnEnable()
    {
        _startTime = Time.unscaledTime;
        _initialCaptureDone = false;
        _currentState = "idle";
        _currentCondition = "";
        _lastStatusPoll = 0f;
        _lastCollectSend = 0f;
    }

    void Update()
    {
        if (emotionClient == null) return;

        // --- Initial quick capture to verify system works ---
        if (doInitialCapture && !_initialCaptureDone && !_requestInFlight)
        {
            if (Time.unscaledTime - _startTime >= initialCaptureSec)
            {
                _initialCaptureDone = true;
                FaceFeatures quickSummary = Summarize(initialCaptureSec);
                string asrText = emotionClient.latestAsrText ?? "";
                string utcIso = DateTime.UtcNow.ToString("o");

                Debug.Log($"[EmotionAggregator] Initial quick capture at {initialCaptureSec}s — verifying system works...");
                StartCoroutine(PostGuarded(utcIso, asrText, quickSummary));
                return;
            }
        }

        // --- Poll iteration status from server ---
        if (Time.unscaledTime - _lastStatusPoll >= statusPollInterval)
        {
            _lastStatusPoll = Time.unscaledTime;
            StartCoroutine(PollIterationStatus());
        }

        // --- During "collecting" state, send data periodically ---
        // BUT skip if condition is "value_only" (save GPU)
        if (_currentState == "collecting" && !_requestInFlight)
        {
            if (_currentCondition == "value_only")
            {
                // Value-only: don't capture or send emotion data
                return;
            }

            if (Time.unscaledTime - _lastCollectSend >= collectSendInterval)
            {
                _lastCollectSend = Time.unscaledTime;

                FaceFeatures summary = Summarize(collectSendInterval);
                string asrText = emotionClient.latestAsrText ?? "";
                string utcIso = DateTime.UtcNow.ToString("o");

                Debug.Log($"[EmotionAggregator] Collecting — sending data to server " +
                          $"(iter={_currentIteration}, condition={_currentCondition}, samples={_buf.Count})");
                StartCoroutine(PostGuarded(utcIso, asrText, summary));
                _sentForThisIteration = true;
            }
        }
    }

    private IEnumerator PollIterationStatus()
    {
        string url = emotionServerUrl + "/iteration_status";
        using (var req = UnityWebRequest.Get(url))
        {
            req.timeout = 5;
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    var status = JsonUtility.FromJson<IterationStatus>(req.downloadHandler.text);
                    string prevState = _currentState;
                    _currentState = status.state ?? "idle";
                    _currentIteration = status.iteration;
                    _currentCondition = status.condition ?? "";

                    // Pass study IDs to EmotionClient for CSV logging
                    if (emotionClient != null)
                    {
                        emotionClient.studyUserId = status.user_id ?? "";
                        emotionClient.studyConditionId = status.condition_id ?? "";
                        emotionClient.studyGroupId = status.group_id ?? "";
                        emotionClient.studyIteration = status.iteration;
                    }

                    // Log state transitions
                    if (prevState != _currentState)
                    {
                        Debug.Log($"[EmotionAggregator] State: {prevState} -> {_currentState} " +
                                  $"(iter={_currentIteration}, condition={_currentCondition})");

                        if (_currentState == "collecting")
                        {
                            _sentForThisIteration = false;
                            _lastCollectSend = Time.unscaledTime;

                            if (_currentCondition == "value_only")
                            {
                                Debug.Log("[EmotionAggregator] Value-only condition — emotion capture DISABLED for this iteration.");
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[EmotionAggregator] Failed to parse iteration_status: {e.Message}");
                }
            }
        }
    }

    [Serializable]
    private class IterationStatus
    {
        public string state;
        public int iteration;
        public float start_time;
        public int logs_count;
        public string user_id;
        public string condition_id;
        public string group_id;
        public int environment_index;
        public string condition;
    }

    public void AddFaceFeatures(FaceFeatures f)
    {
        if (f == null) return;

        _buf.Add(new Sample
        {
            t = Time.unscaledTime,
            f = Copy(f)
        });

        PruneOld();
    }

    public void OnAsrChunk(string asrText, string utcIso)
    {
        if (emotionClient == null) return;
        if (!sendOnAsrChunk) return;

        asrText = asrText ?? "";

        emotionClient.latestAsrText = asrText;
        emotionClient.latestUtcTimestamp = string.IsNullOrEmpty(utcIso) ? DateTime.UtcNow.ToString("o") : utcIso;

        if (!sendEvenIfTextEmpty && string.IsNullOrWhiteSpace(asrText))
            return;

        if (string.IsNullOrEmpty(utcIso))
            utcIso = DateTime.UtcNow.ToString("o");

        FaceFeatures summary = Summarize(windowSec);

        if (_requestInFlight) return;
        StartCoroutine(PostGuarded(utcIso, asrText, summary));
    }

    private IEnumerator PostGuarded(string utcIso, string asrText, FaceFeatures summary)
    {
        _requestInFlight = true;
        yield return StartCoroutine(emotionClient.PostOnce(utcIso, asrText, summary, drivingSession));
        _requestInFlight = false;
    }

    private void PruneOld()
    {
        float cutoff = Time.unscaledTime - Mathf.Max(keepSec, windowSec);
        int removeCount = 0;
        for (int i = 0; i < _buf.Count; i++)
        {
            if (_buf[i].t >= cutoff) break;
            removeCount++;
        }
        if (removeCount > 0) _buf.RemoveRange(0, removeCount);
    }

    private FaceFeatures Summarize(float sec)
    {
        float now = Time.unscaledTime;
        float start = now - sec;

        int n = 0;
        float sumMouth = 0, sumSmile = 0, sumEye = 0, sumBrowR = 0, sumBrowF = 0;
        float sumYaw = 0, sumPitch = 0, sumRoll = 0;
        int blinkCloses = 0;
        bool wasClosed = false;

        for (int i = 0; i < _buf.Count; i++)
        {
            if (_buf[i].t < start) continue;

            var f = _buf[i].f;
            n++;

            sumMouth += f.mouth_open;
            sumSmile += f.smile;
            sumEye += f.eye_open;
            sumBrowR += f.brow_raise;
            sumBrowF += f.brow_furrow;
            sumYaw += f.head_yaw;
            sumPitch += f.head_pitch;
            sumRoll += f.head_roll;

            bool closed = f.eye_open < eyeClosedThreshold;
            if (!wasClosed && closed) blinkCloses++;
            wasClosed = closed;
        }

        var outF = new FaceFeatures();
        if (n <= 0) return outF;

        outF.mouth_open = sumMouth / n;
        outF.smile = sumSmile / n;
        outF.eye_open = sumEye / n;
        outF.brow_raise = sumBrowR / n;
        outF.brow_furrow = sumBrowF / n;
        outF.head_yaw = sumYaw / n;
        outF.head_pitch = sumPitch / n;
        outF.head_roll = sumRoll / n;

        float scaleTo10 = 10f / Mathf.Max(0.1f, sec);
        outF.blink_rate_10s = blinkCloses * scaleTo10;

        return outF;
    }

    private FaceFeatures Copy(FaceFeatures f)
    {
        return new FaceFeatures
        {
            mouth_open = f.mouth_open,
            smile = f.smile,
            brow_raise = f.brow_raise,
            brow_furrow = f.brow_furrow,
            eye_open = f.eye_open,
            head_yaw = f.head_yaw,
            head_pitch = f.head_pitch,
            head_roll = f.head_roll,
            blink_rate_10s = f.blink_rate_10s
        };
    }
}

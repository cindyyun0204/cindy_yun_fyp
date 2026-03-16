using System;
using System.Collections.Generic;
using UnityEngine;

public class EmotionWindowAggregator : MonoBehaviour
{
    [Header("References")]
    public EmotionClient emotionClient;

    [Header("Window")]
    public float windowSec = 9f;
    public float keepSec = 13f;

    [Header("Blink detection")]
    public float eyeClosedThreshold = 0.18f;

    [Header("Send settings")]
    public bool drivingSession = true;
    public bool sendEvenIfTextEmpty = true;
    public bool sendOnTimer = true;
    public float timerIntervalSec = 3f;
    public bool sendOnAsrChunk = true;

    private readonly List<Sample> _buf = new List<Sample>(2048);

    private struct Sample
    {
        public float t;
        public FaceFeatures f;
    }

    private float _nextTimerSendTime = 0f;
    private bool _requestInFlight = false;

    void Awake()
    {
        if (emotionClient == null) emotionClient = GetComponent<EmotionClient>();
        if (emotionClient == null) emotionClient = FindObjectOfType<EmotionClient>();
    }

    void OnEnable()
    {
        _nextTimerSendTime = Time.unscaledTime + Mathf.Max(0.1f, timerIntervalSec);
    }

    void Update()
    {
        if (!sendOnTimer) return;
        if (emotionClient == null) return;

        if (Time.unscaledTime < _nextTimerSendTime) return;
        _nextTimerSendTime = Time.unscaledTime + Mathf.Max(0.1f, timerIntervalSec);

        if (_requestInFlight) return;

        FaceFeatures summary = Summarize(windowSec);

        // Pull latest text from EmotionClient
        string asrText = emotionClient.latestAsrText ?? "";

        if (!sendEvenIfTextEmpty && string.IsNullOrWhiteSpace(asrText))
            return;

        string utcIso = DateTime.UtcNow.ToString("o");
        StartCoroutine(PostGuarded(utcIso, asrText, summary));
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

    private System.Collections.IEnumerator PostGuarded(string utcIso, string asrText, FaceFeatures summary)
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

using System;
using System.Collections;
using System.Globalization;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;

public class LogTailerEmotionBridge : MonoBehaviour
{
    private const string WHISPER_FILE = "transcripts.jsonl";
    private const string FACE_FILE = "face_landmarks_468.jsonl";

    [Header("References")]
    public EmotionWindowAggregator aggregator;

    [Header("Polling")]
    public float pollIntervalSec = 0.1f;

    private string whisperJsonlPath;
    private string faceJsonlPath;

    private long whisperPos = 0;
    private long facePos = 0;

    private WhisperFrame lastWhisper;
    private FaceFrame lastFace;

    [Serializable]
    private class WhisperFrame
    {
        public string utc;
        public float time;
        public string text;
        public string raw;
    }

    [Serializable]
    private class FaceFrame
    {
        public string utc;
        public float unityTime;
        public int faceIndex;
        public FaceLandmark[] landmarks;
    }

    [Serializable]
    private class FaceLandmark
    {
        public int i;
        public float x;
        public float y;
        public float z;
    }

    void Awake()
    {
        // All log inputs live in the same LogsDir
        string logFolder = LogPaths.LogsDir;

        whisperJsonlPath = Path.Combine(logFolder, WHISPER_FILE);
        faceJsonlPath = Path.Combine(logFolder, FACE_FILE);

        if (aggregator == null)
            aggregator = FindObjectOfType<EmotionWindowAggregator>();

        Debug.Log($"[LogTailerEmotionBridge] LogsDir: {logFolder}");
        Debug.Log($"[LogTailerEmotionBridge] Whisper JSONL: {whisperJsonlPath}");
        Debug.Log($"[LogTailerEmotionBridge] Face JSONL: {faceJsonlPath}");
    }

    void Start()
    {
        StartCoroutine(PollLoop());
    }

    IEnumerator PollLoop()
    {
        while (true)
        {
            yield return new WaitForSecondsRealtime(pollIntervalSec);

            TailFace();
            TailWhisper();
        }
    }

    void TailWhisper()
    {
        if (aggregator == null) return;
        if (!File.Exists(whisperJsonlPath)) return;

        bool gotNew = false;

        using (var fs = new FileStream(whisperJsonlPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (var sr = new StreamReader(fs))
        {
            if (fs.Length < whisperPos) whisperPos = 0;
            fs.Seek(whisperPos, SeekOrigin.Begin);

            while (!sr.EndOfStream)
            {
                string line = sr.ReadLine();
                if (string.IsNullOrWhiteSpace(line)) continue;

                try
                {
                    var wf = JsonConvert.DeserializeObject<WhisperFrame>(line);
                    if (wf != null && !string.IsNullOrEmpty(wf.utc))
                    {
                        lastWhisper = wf;
                        gotNew = true;
                    }
                }
                catch
                {
                    continue;
                }
            }

            whisperPos = fs.Position;
        }

        if (gotNew && lastWhisper != null)
        {
            aggregator.OnAsrChunk(lastWhisper.text ?? "", NormalizeUtc(lastWhisper.utc));
        }
    }

    void TailFace()
    {
        if (aggregator == null) return;
        if (!File.Exists(faceJsonlPath)) return;

        bool gotAny = false;

        using (var fs = new FileStream(faceJsonlPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (var sr = new StreamReader(fs))
        {
            if (fs.Length < facePos) facePos = 0;
            fs.Seek(facePos, SeekOrigin.Begin);

            while (!sr.EndOfStream)
            {
                string line = sr.ReadLine();
                if (string.IsNullOrWhiteSpace(line)) continue;

                try
                {
                    var ff = JsonConvert.DeserializeObject<FaceFrame>(line);

                    if (ff != null &&
                        !string.IsNullOrEmpty(ff.utc) &&
                        ff.landmarks != null &&
                        ff.landmarks.Length > 10)
                    {
                        lastFace = ff;
                        gotAny = true;
                    }
                }
                catch
                {
                    continue;
                }
            }

            facePos = fs.Position;
        }

        if (gotAny && lastFace != null)
        {
            var lm = ConvertLandmarks(lastFace.landmarks);
            var features = ExtractFaceFeatures(lm);
            aggregator.AddFaceFeatures(features);
        }
    }

    static string NormalizeUtc(string iso)
    {
        if (TryParseUtc(iso, out var dt))
            return dt.ToString("o");
        return DateTime.UtcNow.ToString("o");
    }

    static bool TryParseUtc(string iso, out DateTime dt)
    {
        return DateTime.TryParse(
            iso,
            null,
            DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal,
            out dt
        );
    }

    static Vector3[] ConvertLandmarks(FaceLandmark[] landmarks)
    {
        if (landmarks == null || landmarks.Length == 0)
            return Array.Empty<Vector3>();

        int maxI = -1;
        for (int k = 0; k < landmarks.Length; k++)
            if (landmarks[k].i > maxI) maxI = landmarks[k].i;

        var arr = new Vector3[maxI + 1];
        for (int k = 0; k < landmarks.Length; k++)
        {
            var lm = landmarks[k];
            if (lm.i < 0 || lm.i >= arr.Length) continue;
            arr[lm.i] = new Vector3(lm.x, lm.y, lm.z);
        }
        return arr;
    }

    static FaceFeatures ExtractFaceFeatures(Vector3[] lm)
    {
        var f = new FaceFeatures();
        if (lm == null) return f;

        float faceWidth = SafeDist(lm, 234, 454);
        if (faceWidth < 1e-6f) faceWidth = 1f;

        float mouthOpen = SafeDist(lm, 13, 14) / faceWidth;
        float mouthWidth = SafeDist(lm, 61, 291) / faceWidth;
        float smile = Mathf.Clamp01((mouthWidth - 0.25f) * 3f);

        float leftEye = SafeDist(lm, 159, 145) / faceWidth;
        float rightEye = SafeDist(lm, 386, 374) / faceWidth;
        float eyeOpen = (leftEye + rightEye) * 0.5f;

        float leftBrow = SafeDist(lm, 105, 159) / faceWidth;
        float rightBrow = SafeDist(lm, 334, 386) / faceWidth;
        float browRaiseRaw = (leftBrow + rightBrow) * 0.5f;

        float browInnerDist = SafeDist(lm, 107, 336) / faceWidth;
        float browFurrow = Mathf.Clamp01((0.23f - browInnerDist) * 10f);

        f.mouth_open = Mathf.Clamp01(mouthOpen * 6f);
        f.smile = smile;
        f.eye_open = Mathf.Clamp01(eyeOpen * 10f);
        f.brow_raise = Mathf.Clamp01((browRaiseRaw - 0.03f) * 20f);
        f.brow_furrow = browFurrow;

        f.head_yaw = 0f;
        f.head_pitch = 0f;
        f.head_roll = 0f;
        f.blink_rate_10s = 0f;

        return f;
    }

    static float SafeDist(Vector3[] lm, int a, int b)
    {
        if (a < 0 || b < 0 || a >= lm.Length || b >= lm.Length) return 0f;
        return (lm[a] - lm[b]).magnitude;
    }
}

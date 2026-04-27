using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

[Serializable]
public class FaceFeatures
{
    public float mouth_open;
    public float smile;
    public float brow_raise;
    public float brow_furrow;
    public float eye_open;
    public float head_yaw;
    public float head_pitch;
    public float head_roll;
    public float blink_rate_10s;
}

[Serializable]
public class EmotionRequest
{
    public string session_id;
    public string t_utc;
    public string asr_text;
    public FaceFeatures face_features;
    public bool driving_session;
    public string face_jpeg_b64;
}

[Serializable]
public class EmotionResponse
{
    public float safety;
    public float naturalness;
    public float progress;
    public string emotion;
    public float valence;
    public float arousal;
    public float confidence;
    public string[] signals;
    public string notes;
}

public class EmotionClient : MonoBehaviour
{
    [Header("Server (Multimodal to Qwen-VL)")]
    public string serverUrl = "http://127.0.0.1:8000/emotion_from_logs";
    public string sessionId = "session-001";

    [Header("Live fields")]
    public string latestAsrText = "";
    public string latestUtcTimestamp = "";
    public FaceFeatures latestFaceFeatures = new FaceFeatures();

    [Header("Last response")]
    public EmotionResponse lastResponse;

    [Header("Debug")]
    public bool logRequests = true;
    public bool logResponses = true;

    [Header("CSV Logging")]
    public bool enableCsvLogging = true;
    public string csvFileName = "emotion_log.csv";

    [Header("Incoming face images (saved for dataset/debug)")]
    public bool saveFaceImages = true;
    [Tooltip("If empty, uses persistentDataPath/Logs/images")]
    public string saveImageDirectory = "";
    public string saveImagePrefix = "face_";

    [Header("Server image preprocess (what gets base64'd to server)")]
    public int serverVisionSize = 256;
    [Range(1, 100)] public int serverVisionJpgQuality = 70;

    // Study IDs — set by EmotionWindowAggregator from iteration_status
    [HideInInspector] public string studyUserId = "";
    [HideInInspector] public string studyConditionId = "";
    [HideInInspector] public string studyGroupId = "";
    [HideInInspector] public int studyIteration = 0;

    // Latest blendshapes received from the MediaPipe runner
    private Dictionary<string, float> _latestBlendshapes = new Dictionary<string, float>();
    private readonly object _blendshapeLock = new object();

    private readonly object _latestFaceLock = new object();
    private byte[] _latestFaceJpgForServer = null;

    private Texture2D _decodeTex;
    private Texture2D _resizeTex;

    private static readonly string[] CsvHeader = new[]
    {
        "t_utc", "session_id", "user_id", "condition_id", "group_id", "iteration",
        "driving_session", "asr_text",
        "mouth_open", "smile", "brow_raise", "brow_furrow", "eye_open",
        "head_yaw", "head_pitch", "head_roll", "blink_rate_10s",
        "blendshape_count",
        "server_url", "http_result", "http_error",
        "safety", "naturalness", "progress",
        "emotion", "valence", "arousal", "confidence",
        "signals", "notes"
    };

    void Awake()
    {
        Debug.Log($"[EmotionClient] LogsDir: {LogPaths.LogsDir}");
        if (enableCsvLogging) CsvLog.Init(csvFileName, CsvHeader);
        EnsureImagesDir();
    }

    private void EnsureImagesDir()
    {
        if (!saveFaceImages) return;
        string dir = string.IsNullOrWhiteSpace(saveImageDirectory) ? LogPaths.ImagesDir : saveImageDirectory;
        try { Directory.CreateDirectory(dir); }
        catch (Exception e) { Debug.LogWarning($"[EmotionClient] Failed to create image dir '{dir}': {e.Message}"); }
    }

    public void SetLatestFaceFeatures(string utcIso, FaceFeatures features)
    {
        latestUtcTimestamp = string.IsNullOrEmpty(utcIso) ? DateTime.UtcNow.ToString("o") : utcIso;
        latestFaceFeatures = features ?? new FaceFeatures();
    }

    public void SetLatestBlendshapes(Dictionary<string, float> shapes)
    {
        if (shapes == null) return;
        lock (_blendshapeLock)
        {
            _latestBlendshapes = new Dictionary<string, float>(shapes);
        }
    }

    public void SetLatestFaceImage(byte[] bytes, string mimeType, string utcIso = null)
    {
        if (bytes == null || bytes.Length == 0) return;

        if (saveFaceImages)
            SaveIncomingBytesToDisk(bytes, mimeType, utcIso);

        byte[] jpg = ConvertToJpgSquare(bytes, serverVisionSize, serverVisionJpgQuality);
        if (jpg == null || jpg.Length == 0) return;

        lock (_latestFaceLock)
        {
            _latestFaceJpgForServer = jpg;
        }
    }

    private void SaveIncomingBytesToDisk(byte[] bytes, string mimeType, string utcIso)
    {
        string dir = string.IsNullOrWhiteSpace(saveImageDirectory) ? LogPaths.ImagesDir : saveImageDirectory;

        string ext = "jpg";
        if (!string.IsNullOrEmpty(mimeType) && mimeType.Contains("png")) ext = "png";

        string ts = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff'Z'");
        if (!string.IsNullOrEmpty(utcIso))
            ts = utcIso.Replace(":", "").Replace("-", "").Replace(".", "_");

        string safeSession = string.IsNullOrEmpty(sessionId) ? "session" : sessionId.Replace(":", "_").Replace("/", "_");
        string fileName = $"{saveImagePrefix}{safeSession}_{ts}.{ext}";
        string path = Path.Combine(dir, fileName);

        try
        {
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(path, bytes);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[EmotionClient] Failed to save image '{path}': {e.Message}");
        }
    }

    private byte[] ConvertToJpgSquare(byte[] bytes, int size, int jpgQuality)
    {
        try
        {
            if (_decodeTex == null)
                _decodeTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);

            if (!_decodeTex.LoadImage(bytes, markNonReadable: false))
                return null;

            if (_resizeTex == null || _resizeTex.width != size || _resizeTex.height != size)
                _resizeTex = new Texture2D(size, size, TextureFormat.RGBA32, false);

            var rt = RenderTexture.GetTemporary(size, size, 0, RenderTextureFormat.ARGB32);
            var prev = RenderTexture.active;

            float srcAspect = (float)_decodeTex.width / Mathf.Max(1, _decodeTex.height);
            Vector2 scale = Vector2.one;
            Vector2 offset = Vector2.zero;

            if (srcAspect > 1f) { scale.x = 1f / srcAspect; offset.x = (1f - scale.x) * 0.5f; }
            else if (srcAspect < 1f) { scale.y = srcAspect; offset.y = (1f - scale.y) * 0.5f; }

            Graphics.Blit(_decodeTex, rt, scale, offset);
            RenderTexture.active = rt;
            _resizeTex.ReadPixels(new UnityEngine.Rect(0, 0, size, size), 0, 0);
            _resizeTex.Apply(false, false);
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(rt);

            return _resizeTex.EncodeToJPG(Mathf.Clamp(jpgQuality, 1, 100));
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[EmotionClient] ConvertToJpgSquare failed: {e.Message}");
            return null;
        }
    }

    public IEnumerator PostOnce(string utcIso, string asr, FaceFeatures features, bool drivingSession = true)
    {
        utcIso   = string.IsNullOrEmpty(utcIso) ? DateTime.UtcNow.ToString("o") : utcIso;
        asr      = asr ?? "";
        features = features ?? new FaceFeatures();

        latestUtcTimestamp = utcIso;
        latestAsrText      = asr;
        latestFaceFeatures = features;

        string faceB64 = "";
        lock (_latestFaceLock)
        {
            if (_latestFaceJpgForServer != null && _latestFaceJpgForServer.Length > 0)
                faceB64 = Convert.ToBase64String(_latestFaceJpgForServer);
        }

        Dictionary<string, float> blendshapesSnapshot;
        lock (_blendshapeLock)
        {
            blendshapesSnapshot = new Dictionary<string, float>(_latestBlendshapes);
        }

        string json = BuildRequestJson(utcIso, asr, features, drivingSession, faceB64, blendshapesSnapshot);

        if (logRequests)
            Debug.Log($"[EmotionClient] POST /emotion_from_logs t={utcIso} " +
                      $"asrLen={asr.Length} img={(faceB64.Length > 0 ? "yes" : "no")} " +
                      $"blendshapes={blendshapesSnapshot.Count} " +
                      $"user={studyUserId} iter={studyIteration}");

        string httpResult = "UNKNOWN";
        string httpError  = "";
        EmotionResponse parsed = null;

        using (var www = new UnityWebRequest(serverUrl, "POST"))
        {
            www.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = 120;

            yield return www.SendWebRequest();

            string respJson = www.downloadHandler != null ? www.downloadHandler.text : "";

            if (www.result != UnityWebRequest.Result.Success)
            {
                httpResult = "FAIL";
                httpError  = www.error ?? "";
                Debug.LogWarning($"[EmotionClient] POST failed: {www.error} body={respJson}");
                WriteCsvRow(utcIso, asr, features, drivingSession, blendshapesSnapshot, httpResult, httpError, null);
                yield break;
            }

            httpResult = "OK";
            try
            {
                parsed = JsonUtility.FromJson<EmotionResponse>(respJson);
                lastResponse = parsed;

                if (logResponses && parsed != null)
                    Debug.Log($"[EmotionClient] Emotion={parsed.emotion} conf={parsed.confidence:0.00} " +
                              $"safety={parsed.safety:0.00} nat={parsed.naturalness:0.00} prog={parsed.progress:0.00}");
            }
            catch (Exception e)
            {
                httpResult = "PARSE_FAIL";
                httpError  = e.Message ?? "parse error";
                Debug.LogWarning($"[EmotionClient] Failed to parse response: {e.Message}\nRaw: {respJson}");
            }

            WriteCsvRow(utcIso, asr, features, drivingSession, blendshapesSnapshot, httpResult, httpError, parsed);
        }
    }

    // ── JSON builder ────────────────────────────────────────────────────────

    private string BuildRequestJson(
        string utcIso, string asr, FaceFeatures f, bool driving,
        string faceB64, Dictionary<string, float> blendshapes)
    {
        var sb = new StringBuilder();
        sb.Append('{');
        sb.Append($"\"session_id\":\"{EscJ(sessionId)}\",");
        sb.Append($"\"t_utc\":\"{EscJ(utcIso)}\",");
        sb.Append($"\"asr_text\":\"{EscJ(asr)}\",");
        sb.Append($"\"driving_session\":{(driving ? "true" : "false")},");
        sb.Append($"\"face_jpeg_b64\":\"{faceB64}\",");

        sb.Append("\"face_features\":{");
        sb.Append($"\"mouth_open\":{f.mouth_open.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.Append($"\"smile\":{f.smile.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.Append($"\"brow_raise\":{f.brow_raise.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.Append($"\"brow_furrow\":{f.brow_furrow.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.Append($"\"eye_open\":{f.eye_open.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.Append($"\"head_yaw\":{f.head_yaw.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.Append($"\"head_pitch\":{f.head_pitch.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.Append($"\"head_roll\":{f.head_roll.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},");
        sb.Append($"\"blink_rate_10s\":{f.blink_rate_10s.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)}");
        sb.Append("},");

        sb.Append("\"blendshapes\":{");
        bool first = true;
        foreach (var kvp in blendshapes)
        {
            if (!first) sb.Append(',');
            sb.Append($"\"{EscJ(kvp.Key)}\":{kvp.Value.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)}");
            first = false;
        }
        sb.Append('}');

        sb.Append('}');
        return sb.ToString();
    }

    private static string EscJ(string s) =>
        (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");

    // ── CSV logging ─────────────────────────────────────────────────────────

    private void WriteCsvRow(
        string utcIso, string asr, FaceFeatures f, bool driving,
        Dictionary<string, float> blendshapes,
        string httpResult, string httpError, EmotionResponse parsed)
    {
        if (!enableCsvLogging) return;

        string signalsJoined = (parsed?.signals != null) ? string.Join("|", parsed.signals) : "";
        CsvLog.Init(csvFileName, CsvHeader);

        CsvLog.AppendRow(
            utcIso,
            sessionId,
            studyUserId,
            studyConditionId,
            studyGroupId,
            studyIteration.ToString(),
            driving ? "true" : "false",
            asr,
            f.mouth_open.ToString("F6"),
            f.smile.ToString("F6"),
            f.brow_raise.ToString("F6"),
            f.brow_furrow.ToString("F6"),
            f.eye_open.ToString("F6"),
            f.head_yaw.ToString("F6"),
            f.head_pitch.ToString("F6"),
            f.head_roll.ToString("F6"),
            f.blink_rate_10s.ToString("F6"),
            blendshapes.Count.ToString(),
            serverUrl,
            httpResult,
            httpError ?? "",
            parsed != null ? parsed.safety.ToString("F6")       : "",
            parsed != null ? parsed.naturalness.ToString("F6")  : "",
            parsed != null ? parsed.progress.ToString("F6")     : "",
            parsed != null ? (parsed.emotion ?? "")             : "",
            parsed != null ? parsed.valence.ToString("F6")      : "",
            parsed != null ? parsed.arousal.ToString("F6")      : "",
            parsed != null ? parsed.confidence.ToString("F6")   : "",
            signalsJoined,
            parsed != null ? (parsed.notes ?? "")               : ""
        );
    }
}
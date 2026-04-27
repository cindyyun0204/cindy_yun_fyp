using System;
using System.IO;
using System.Text;
using System.Collections.Concurrent;
using UnityEngine;

public class FaceLandmarksJsonlLogger : MonoBehaviour
{
    [Header("Output")]
    public string outputDirectory = "";
    public string logFileName = "face_landmarks_468.jsonl";

    public bool useUtcTime = true;
    public bool enablePushMode = true;

    private string _fullPath;
    private readonly StringBuilder _sb = new StringBuilder(64 * 1024);

    private readonly ConcurrentQueue<QueuedFrame> _queue = new();

    private struct QueuedFrame
    {
        public Vector3[] Landmarks;
        public int FaceIndex;
        public string UtcIso;
    }

    void Awake()
    {
        ResolvePath();
    }

    void Start()
    {
        Debug.Log($"[FaceLandmarksJsonlLogger] Writing to: {_fullPath}");
    }

    private void ResolvePath()
    {
        string dir = string.IsNullOrWhiteSpace(outputDirectory) ? LogPaths.LogsDir : outputDirectory;
        dir = Path.GetFullPath(dir);
        Directory.CreateDirectory(dir);

        if (string.IsNullOrWhiteSpace(logFileName))
            logFileName = "face_landmarks_468.jsonl";

        _fullPath = Path.Combine(dir, logFileName);
    }

    void Update()
    {
        while (_queue.TryDequeue(out var f))
        {
            float unityTime = Time.unscaledTime;
            string line = BuildJsonLine(f.UtcIso, unityTime, f.FaceIndex, f.Landmarks);
            AppendJsonl(line);
        }
    }

    public void LogLandmarks(Vector3[] landmarks, int faceIndex = 0, string utcIso = null)
    {
        if (!enablePushMode) return;
        if (landmarks == null || landmarks.Length == 0) return;

        var copy = (Vector3[])landmarks.Clone();

        string utc = useUtcTime
            ? (string.IsNullOrEmpty(utcIso) ? DateTime.UtcNow.ToString("o") : utcIso)
            : "";

        _queue.Enqueue(new QueuedFrame
        {
            Landmarks = copy,
            FaceIndex = faceIndex,
            UtcIso = utc
        });
    }

    private void AppendJsonl(string line)
    {
        try
        {
            using (var fs = new FileStream(_fullPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
            using (var sw = new StreamWriter(fs, Encoding.UTF8))
            {
                sw.WriteLine(line);
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[FaceLandmarksJsonlLogger] Write failed: {e.Message}");
        }
    }

    private string BuildJsonLine(string utc, float unityTime, int faceIndex, Vector3[] landmarks)
    {
        _sb.Clear();

        _sb.Append('{');

        _sb.Append("\"utc\":");
        if (useUtcTime) _sb.Append('\"').Append(EscapeJson(utc)).Append('\"');
        else _sb.Append("\"\"");

        _sb.Append(",\"unityTime\":").Append(unityTime.ToString("0.######"));
        _sb.Append(",\"faceIndex\":").Append(faceIndex);

        _sb.Append(",\"landmarks\":[");
        int n = landmarks.Length;

        for (int i = 0; i < n; i++)
        {
            Vector3 v = landmarks[i];
            if (i > 0) _sb.Append(',');

            _sb.Append('{');
            _sb.Append("\"i\":").Append(i);
            _sb.Append(",\"x\":").Append(v.x.ToString("0.######"));
            _sb.Append(",\"y\":").Append(v.y.ToString("0.######"));
            _sb.Append(",\"z\":").Append(v.z.ToString("0.######"));
            _sb.Append('}');
        }

        _sb.Append(']');
        _sb.Append('}');

        return _sb.ToString();
    }

    private static string EscapeJson(string s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
    }
}

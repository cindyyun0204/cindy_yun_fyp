using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

[Serializable]
public class OfficialMediaPipeRequest
{
    public string image_b64;
    public long timestamp_ms;
}

[Serializable]
public class OfficialMediaPipeLandmark
{
    public int i;
    public float x;
    public float y;
    public float z;
}

// Single blendshape entry for JsonUtility array deserialisation.
// JsonUtility can't deserialise Dictionary directly, so the service
// returns blendshapes as a plain JSON object and we parse it manually.
[Serializable]
public class OfficialMediaPipeResponse
{
    public bool ok;
    public bool face_detected;
    public long timestamp_ms;
    public OfficialMediaPipeLandmark[] landmarks;
    public string error;

    // Populated after deserialization by ParseBlendshapes()
    [NonSerialized]
    public Dictionary<string, float> blendshapes = new Dictionary<string, float>();
}

public class OfficialMediaPipeClient : MonoBehaviour
{
    [Header("Official MediaPipe service")]
    public string serviceUrl = "http://127.0.0.1:8010/mediapipe_face";
    public int timeoutSec = 60;
    public bool logRequests = false;
    public bool logResponses = false;

    void Awake()
    {
        var envUrl = Environment.GetEnvironmentVariable("MEDIAPIPE_BASE_URL");
        if (!string.IsNullOrEmpty(envUrl))
            serviceUrl = envUrl.TrimEnd('/') + "/mediapipe_face";
    }

    public IEnumerator DetectFromImageBytes(byte[] imageBytes, Action<OfficialMediaPipeResponse> onDone)
    {
        if (imageBytes == null || imageBytes.Length == 0)
        {
            onDone?.Invoke(new OfficialMediaPipeResponse
            {
                ok = false, face_detected = false, error = "No image bytes"
            });
            yield break;
        }

        var reqObj = new OfficialMediaPipeRequest
        {
            image_b64    = Convert.ToBase64String(imageBytes),
            timestamp_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
        };

        string json = JsonUtility.ToJson(reqObj);
        byte[] body = Encoding.UTF8.GetBytes(json);

        if (logRequests)
            Debug.Log($"[OfficialMediaPipeClient] POST {serviceUrl} bytes={imageBytes.Length}");

        using (var req = new UnityWebRequest(serviceUrl, "POST"))
        {
            req.uploadHandler   = new UploadHandlerRaw(body);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = timeoutSec;

            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                if (logResponses)
                    Debug.LogWarning($"[OfficialMediaPipeClient] HTTP FAIL: {req.error}\n{req.downloadHandler?.text}");

                onDone?.Invoke(new OfficialMediaPipeResponse
                {
                    ok = false, face_detected = false,
                    error = req.error ?? "request failed"
                });
                yield break;
            }

            string respText = req.downloadHandler != null ? req.downloadHandler.text : "";

            OfficialMediaPipeResponse parsed = null;
            try
            {
                parsed = JsonUtility.FromJson<OfficialMediaPipeResponse>(respText);

                // JsonUtility can't read Dictionary — parse blendshapes manually
                if (parsed != null)
                    parsed.blendshapes = ParseBlendshapesFromJson(respText);
            }
            catch (Exception e)
            {
                onDone?.Invoke(new OfficialMediaPipeResponse
                {
                    ok = false, face_detected = false,
                    error = "JSON parse failed: " + e.Message
                });
                yield break;
            }

            if (parsed == null)
            {
                onDone?.Invoke(new OfficialMediaPipeResponse
                {
                    ok = false, face_detected = false, error = "Empty response"
                });
                yield break;
            }

            if (logResponses)
                Debug.Log($"[OfficialMediaPipeClient] ok={parsed.ok} face={parsed.face_detected} " +
                          $"landmarks={(parsed.landmarks != null ? parsed.landmarks.Length : 0)} " +
                          $"blendshapes={parsed.blendshapes.Count}");

            onDone?.Invoke(parsed);
        }
    }

    /// <summary>
    /// Minimal JSON parser for the flat "blendshapes": { "key": 0.123, ... } object.
    /// Avoids a Newtonsoft dependency in this file.
    /// </summary>
    private static Dictionary<string, float> ParseBlendshapesFromJson(string json)
    {
        var result = new Dictionary<string, float>();
        if (string.IsNullOrEmpty(json)) return result;

        // Find "blendshapes": { ... }
        int bsIdx = json.IndexOf("\"blendshapes\"", StringComparison.Ordinal);
        if (bsIdx < 0) return result;

        int openBrace = json.IndexOf('{', bsIdx);
        if (openBrace < 0) return result;

        int closeBrace = json.IndexOf('}', openBrace);
        if (closeBrace < 0) return result;

        string inner = json.Substring(openBrace + 1, closeBrace - openBrace - 1);
        // inner is now: "browDownLeft": 0.012, "jawOpen": 0.45, ...
        string[] pairs = inner.Split(',');

        foreach (string pair in pairs)
        {
            string p = pair.Trim();
            if (string.IsNullOrEmpty(p)) continue;

            int colon = p.IndexOf(':');
            if (colon < 0) continue;

            string key   = p.Substring(0, colon).Trim().Trim('"');
            string valStr = p.Substring(colon + 1).Trim();

            if (float.TryParse(valStr,
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out float val))
            {
                result[key] = val;
            }
        }

        return result;
    }
}

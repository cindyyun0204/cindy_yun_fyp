using System;
using System.Collections;
using System.IO;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;

public class WhisperXClient : MonoBehaviour
{
    [Header("Server")]
    public string serverUrl = "http://127.0.0.1:8000/transcribe";

    [Header("UI")]
    public TMP_Text transcriptText;

    [Header("Recording")]
    public int sampleRate = 16000;
    public float recordSeconds = 3f;

    [Header("Integration")]
    public EmotionClient emotionClient;
    public EmotionWindowAggregator aggregator;

    [Header("Logging")]
    public bool enableTranscriptLogging = true;

    private string micDevice;

    private string LogsDir => LogPaths.LogsDir;
    private string TranscriptPath => Path.Combine(LogsDir, "transcripts.jsonl");

    private void Start()
    {
        var envUrl = Environment.GetEnvironmentVariable("EMOTION_SERVER_BASE_URL");
        if (!string.IsNullOrEmpty(envUrl)) serverUrl = envUrl.TrimEnd('/') + "/transcribe";
    
        if (transcriptText != null) transcriptText.text = "";


        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("No microphone found.");
            return;
        }

        if (emotionClient == null) emotionClient = FindObjectOfType<EmotionClient>();
        if (aggregator == null) aggregator = FindObjectOfType<EmotionWindowAggregator>();

        micDevice = Microphone.devices[0];
        Directory.CreateDirectory(LogsDir);

        StartCoroutine(LoopRecordAndSend());
    }

    private IEnumerator LoopRecordAndSend()
    {
        while (true)
        {
            int lengthSec = Mathf.CeilToInt(recordSeconds);
            AudioClip clip = Microphone.Start(micDevice, false, lengthSec, sampleRate);

            yield return new WaitForSeconds(recordSeconds);

            Microphone.End(micDevice);

            if (clip == null) continue;

            byte[] wav = AudioClipToWav(clip);
            yield return SendToWhisperX(wav);
        }
    }

    private IEnumerator SendToWhisperX(byte[] wavBytes)
    {
        WWWForm form = new WWWForm();
        form.AddBinaryData("file", wavBytes, "chunk.wav", "audio/wav");

        using (UnityWebRequest req = UnityWebRequest.Post(serverUrl, form))
        {
            req.timeout = 120;
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError(
                    "[WhisperXClient] HTTP " + req.responseCode +
                    "\nError: " + req.error +
                    "\nResponse:\n" + req.downloadHandler.text
                );
                yield break;
            }


            string json = req.downloadHandler.text;
            string chunkText = ExtractTextField(json);

            if (string.IsNullOrWhiteSpace(chunkText))
                yield break;

            if (transcriptText != null)
            {
                if (transcriptText.text.Length > 0) transcriptText.text += "\n";
                transcriptText.text += chunkText;
            }

            string utc = DateTime.UtcNow.ToString("o");

            // Update EmotionClient's latest ASR
            if (emotionClient != null)
            {
                emotionClient.latestAsrText = chunkText;
                emotionClient.latestUtcTimestamp = utc;
            }

            // Trigger aggregator send-on-ASR
            if (aggregator != null)
            {
                aggregator.OnAsrChunk(chunkText, utc);
            }

            if (enableTranscriptLogging)
                AppendTranscriptChunkJsonl(chunkText, json, utc);
        }
    }

    private string ExtractTextField(string json)
    {
        const string key = "\"text\":";
        int idx = json.IndexOf(key, StringComparison.OrdinalIgnoreCase);
        if (idx < 0) return "";

        int start = json.IndexOf('"', idx + key.Length);
        if (start < 0) return "";
        start++;

        int end = json.IndexOf('"', start);
        if (end < 0) return "";

        return json.Substring(start, end - start).Replace("\\n", "\n");
    }

    private byte[] AudioClipToWav(AudioClip clip)
    {
        float[] samples = new float[clip.samples * clip.channels];
        clip.GetData(samples, 0);

        short[] pcm = new short[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            float s = Mathf.Clamp(samples[i], -1f, 1f);
            pcm[i] = (short)(s * 32767);
        }

        byte[] pcmBytes = new byte[pcm.Length * 2];
        Buffer.BlockCopy(pcm, 0, pcmBytes, 0, pcmBytes.Length);

        return BuildWav(pcmBytes, clip.channels, clip.frequency);
    }

    private byte[] BuildWav(byte[] pcmData, int channels, int hz)
    {
        int byteRate = hz * channels * 2;
        int blockAlign = channels * 2;

        using (var ms = new MemoryStream(44 + pcmData.Length))
        using (var bw = new BinaryWriter(ms))
        {
            bw.Write(Encoding.ASCII.GetBytes("RIFF"));
            bw.Write(36 + pcmData.Length);
            bw.Write(Encoding.ASCII.GetBytes("WAVE"));

            bw.Write(Encoding.ASCII.GetBytes("fmt "));
            bw.Write(16);
            bw.Write((short)1);
            bw.Write((short)channels);
            bw.Write(hz);
            bw.Write(byteRate);
            bw.Write((short)blockAlign);
            bw.Write((short)16);

            bw.Write(Encoding.ASCII.GetBytes("data"));
            bw.Write(pcmData.Length);
            bw.Write(pcmData);

            return ms.ToArray();
        }
    }

    private void AppendTranscriptChunkJsonl(string chunkText, string rawJson, string utc)
    {
        Directory.CreateDirectory(LogsDir);

        var entry = new TranscriptChunkEntry
        {
            utc = utc,
            unityTime = Time.unscaledTime,
            text = chunkText,
            raw = rawJson
        };

        string line = JsonUtility.ToJson(entry);

        try
        {
            File.AppendAllText(TranscriptPath, line + "\n");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[WhisperXClient] Failed to append transcripts.jsonl: {e.Message}");
        }
    }

    [Serializable]
    private class TranscriptChunkEntry
    {
        public string utc;
        public float unityTime;
        public string text;
        public string raw;
    }
}

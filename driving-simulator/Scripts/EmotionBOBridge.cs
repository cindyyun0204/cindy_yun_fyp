using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;
using BOforUnity;
using BOforUnity.Scripts;

public class EmotionBOBridge : MonoBehaviour
{
    [Header("Emotion Server")]
    public string emotionServer = "http://127.0.0.1:8000";

    [Header("References (auto-found at blend time)")]
    public BoForUnityManager boManager;

    [Header("Blending (only used in 'combination' condition)")]
    [Range(0f, 1f)]
    public float emotionWeight = 0.5f;

    // Last fetched emotion result
    private EmotionResult lastEmotionResult = null;
    private bool iterationEndInFlight = false;

    // Latest questionnaire values — set by WriteResults so unified CSV gets fresh data
    [HideInInspector] public float[] latestQuestionnaireSafety = new float[0];
    [HideInInspector] public float[] latestQuestionnaireNaturalness = new float[0];
    [HideInInspector] public float[] latestQuestionnaireProgress = new float[0];

    // CSV paths
    private static string rawCsvPath;
    private static string blendCsvPath;
    private static string unifiedCsvPath;
    private static bool csvInitialized = false;
    private static bool unifiedCsvInitialized = false;

    void Awake()
    {
        if (!csvInitialized)
        {
            string logDir = Path.Combine(Application.persistentDataPath, "Logs");
            if (!Directory.Exists(logDir))
                Directory.CreateDirectory(logDir);

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            rawCsvPath   = Path.Combine(logDir, $"emotion_raw_{timestamp}.csv");
            blendCsvPath = Path.Combine(logDir, $"emotion_blended_{timestamp}.csv");

            File.WriteAllText(rawCsvPath,
                "UserID,ConditionID,GroupID,CurrentIteration,Environment,Condition,Timestamp," +
                "Emotion,Valence,Arousal,Confidence," +
                "Safety,Naturalness,Progress\n");

            File.WriteAllText(blendCsvPath,
                "UserID,ConditionID,GroupID,CurrentIteration,Environment,Condition,Timestamp," +
                "Emotion,Valence,Arousal,Confidence," +
                "Emotion_Safety,Emotion_Naturalness,Emotion_Progress," +
                "Original_Safety1,Original_Safety2,Original_Safety3,Original_Safety4," +
                "Original_Naturalness,Original_Progress," +
                "Final_Safety1,Final_Safety2,Final_Safety3,Final_Safety4," +
                "Final_Naturalness,Final_Progress," +
                "Emotion_Weight,BlendMode\n");

            csvInitialized = true;
            Debug.Log($"[EmotionBridge] Raw CSV:     {rawCsvPath}");
            Debug.Log($"[EmotionBridge] Blended CSV: {blendCsvPath}");
        }

        var envUrl = Environment.GetEnvironmentVariable("EMOTION_SERVER_BASE_URL");
        if (!string.IsNullOrEmpty(envUrl)) emotionServer = envUrl.TrimEnd('/');
    }

    private void InitUnifiedCsv(string userId, string conditionId)
    {
        if (unifiedCsvInitialized) return;

        string logDir = Path.Combine(Application.persistentDataPath, "Logs");
        if (!Directory.Exists(logDir))
            Directory.CreateDirectory(logDir);

        string safeUserId = string.IsNullOrEmpty(userId) ? "unknown" : userId;
        string safeConditionId = string.IsNullOrEmpty(conditionId) ? "unknown" : conditionId;
        unifiedCsvPath = Path.Combine(logDir, $"results_user{safeUserId}_condition{safeConditionId}.csv");

        if (!File.Exists(unifiedCsvPath))
        {
            File.WriteAllText(unifiedCsvPath,
                "UserID,ConditionID,GroupID,CurrentIteration,Environment,Condition,Timestamp," +
                "Questionnaire_Safety1,Questionnaire_Safety2,Questionnaire_Safety3,Questionnaire_Safety4," +
                "Questionnaire_Naturalness,Questionnaire_Progress," +
                "Emotion,Emotion_Valence,Emotion_Arousal,Emotion_Confidence," +
                "Emotion_Safety,Emotion_Naturalness,Emotion_Progress," +
                "Final_Safety1,Final_Safety2,Final_Safety3,Final_Safety4," +
                "Final_Naturalness,Final_Progress," +
                "Emotion_Weight,BlendMode," +
                "Speed,Distance,Braking\n");
        }

        unifiedCsvInitialized = true;
        Debug.Log($"[EmotionBridge] Unified CSV: {unifiedCsvPath}");
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    private (string userId, string conditionId, string groupId, int iteration) GetStudyInfo()
    {
        var mgr = GetBoManager();
        if (mgr != null)
            return (mgr.userId, mgr.conditionId, mgr.groupId, mgr.currentIteration);
        return ("?", "?", "?", 0);
    }

    private BoForUnityManager GetBoManager()
    {
        if (boManager != null) return boManager;
        try
        {
            var go = GameObject.FindGameObjectWithTag("BOforUnityManager");
            if (go != null)
                boManager = go.GetComponent<BoForUnityManager>();
        }
        catch { }
        return boManager;
    }

    private int GetCurrentEnvironmentIndex()
    {
        try
        {
            var loopForQt = GameObject.FindGameObjectWithTag("LoopForQT")
                ?.GetComponent<LoopForQT>();
            return loopForQt != null ? loopForQt.currentEnvironmentIndex : 0;
        }
        catch { return 0; }
    }

    private string GetCurrentCondition()
    {
        var info = GetStudyInfo();
        return StudyConditionManager.GetCondition(info.conditionId);
    }

    private string BuildSignalJson()
    {
        var info = GetStudyInfo();
        int envIdx = GetCurrentEnvironmentIndex();
        string condition = StudyConditionManager.GetCondition(info.conditionId);

        return JsonConvert.SerializeObject(new
        {
            iteration = info.iteration,
            user_id = info.userId,
            condition_id = info.conditionId,
            group_id = info.groupId,
            environment_index = envIdx,
            condition = condition
        });
    }

    private float GetParameterValue(string key)
    {
        var mgr = GetBoManager();
        if (mgr == null) return 0f;
        foreach (var p in mgr.parameters)
        {
            if (p.key == key) return p.value.Value;
        }
        return 0f;
    }

    // ========================================================================
    // ITERATION SIGNALS
    // ========================================================================

    public void SignalIterationStart()
    {
        string condition = GetCurrentCondition();

        // For value_only: still signal so server knows, but emotion system won't capture
        StartCoroutine(PostSignal("/iteration_start"));
    }

    public void SignalIterationEnd()
    {
        string condition = GetCurrentCondition();

        if (condition == StudyConditionManager.VALUE_ONLY)
        {
            // Value-only: don't ask server to process emotion (save GPU)
            Debug.Log("[EmotionBridge] Value-only condition — skipping iteration_end (no emotion processing).");
            return;
        }

        StartCoroutine(PostIterationEnd());
    }

    private IEnumerator PostSignal(string endpoint)
    {
        string url = emotionServer + endpoint;
        string json = BuildSignalJson();

        Debug.Log($"[EmotionBridge] Requesting URL: {url}");

        using (var req = new UnityWebRequest(url, "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = 10;

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
                Debug.Log($"[EmotionBridge] {endpoint} signaled OK: {req.downloadHandler.text}");
            else
                Debug.LogWarning($"[EmotionBridge] {endpoint} failed: {req.error}");
        }
    }

    private IEnumerator PostIterationEnd()
    {
        iterationEndInFlight = true;
        string url = emotionServer + "/iteration_end";
        string json = BuildSignalJson();

        Debug.Log($"[EmotionBridge] Requesting URL: {url}");

        using (var req = new UnityWebRequest(url, "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = 120;

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                string respText = req.downloadHandler.text;
                Debug.Log($"[EmotionBridge] /iteration_end response: {respText}");

                try
                {
                    var parsed = JsonConvert.DeserializeObject<EmotionResult>(respText);
                    if (parsed != null && parsed.confidence > 0.1)
                    {
                        lastEmotionResult = parsed;
                        LogRawEmotion(parsed);
                        Debug.Log($"[EmotionBridge] Emotion ready: {parsed.emotion} " +
                                  $"S:{parsed.safety:F2} N:{parsed.naturalness:F2} P:{parsed.progress:F2}");
                    }
                    else
                    {
                        lastEmotionResult = null;
                        Debug.LogWarning("[EmotionBridge] iteration_end returned low confidence or no data.");
                    }
                }
                catch (Exception e)
                {
                    lastEmotionResult = null;
                    Debug.LogWarning($"[EmotionBridge] Failed to parse iteration_end response: {e.Message}");
                }
            }
            else
            {
                lastEmotionResult = null;
                Debug.LogWarning($"[EmotionBridge] /iteration_end failed: {req.error}");
            }
        }
        iterationEndInFlight = false;
    }

    private void LogRawEmotion(EmotionResult e)
    {
        var info = GetStudyInfo();
        int envIdx = GetCurrentEnvironmentIndex();
        string condition = GetCurrentCondition();
        var ci = CultureInfo.InvariantCulture;

        string rawLine = string.Join(",",
            info.userId, info.conditionId, info.groupId,
            info.iteration.ToString(ci), envIdx.ToString(ci), condition,
            DateTime.Now.ToString("O"),
            e.emotion,
            ((float)e.valence).ToString("F4", ci),
            ((float)e.arousal).ToString("F4", ci),
            ((float)e.confidence).ToString("F4", ci),
            ((float)e.safety).ToString("F4", ci),
            ((float)e.naturalness).ToString("F4", ci),
            ((float)e.progress).ToString("F4", ci)
        );
        File.AppendAllText(rawCsvPath, rawLine + "\n");
    }

    // ========================================================================
    // STORE QUESTIONNAIRE VALUES — called from WriteResults
    // ========================================================================

    /// <summary>
    /// Called from WriteResults in QTQuestionnaireManager to store the actual
    /// questionnaire responses for the unified CSV. This ensures we log the
    /// real values even in environment 1 where BO objectives may not update.
    /// </summary>
    public void StoreQuestionnaireValues(Dictionary<string, List<float>> values)
    {
        if (values.ContainsKey("Safety"))
            latestQuestionnaireSafety = values["Safety"].ToArray();
        if (values.ContainsKey("Naturalness"))
            latestQuestionnaireNaturalness = values["Naturalness"].ToArray();
        if (values.ContainsKey("Progress"))
            latestQuestionnaireProgress = values["Progress"].ToArray();

        Debug.Log($"[EmotionBridge] Stored questionnaire values — " +
                  $"Safety=[{string.Join(",", latestQuestionnaireSafety)}] " +
                  $"Nat=[{string.Join(",", latestQuestionnaireNaturalness)}] " +
                  $"Prog=[{string.Join(",", latestQuestionnaireProgress)}]");
    }

    // ========================================================================
    // APPLY EMOTION TO BO — condition-aware
    // ========================================================================

    /// <summary>
    /// Called from EndOfQt after questionnaire finishes, before OptimizationStart.
    /// Behavior depends on condition:
    ///   llm_only:    Emotion values go DIRECTLY into BO objectives (replace questionnaire values)
    ///   value_only:  Nothing happens here (questionnaire already in BO, emotion skipped)
    ///   combination: Emotion values BLENDED with questionnaire values using emotionWeight
    /// </summary>
    public void ApplyEmotionToObjectives()
    {
        string condition = GetCurrentCondition();

        if (condition == StudyConditionManager.VALUE_ONLY)
        {
            Debug.Log("[EmotionBridge] Value-only — emotion not applied to BO. Writing unified log.");
            WriteUnifiedRow(null, "value_only_direct");
            return;
        }

        StartCoroutine(WaitAndApply());
    }

    private IEnumerator WaitAndApply()
    {
        // Wait for iteration_end to finish if still in flight
        if (iterationEndInFlight)
        {
            Debug.Log("[EmotionBridge] Waiting for iteration_end response...");
            float waitStart = Time.unscaledTime;
            while (iterationEndInFlight && (Time.unscaledTime - waitStart) < 120f)
                yield return null;

            if (iterationEndInFlight)
            {
                Debug.LogWarning("[EmotionBridge] Timed out waiting for iteration_end. Skipping.");
                WriteUnifiedRow(null, "timeout");
                yield break;
            }
        }

        // Fallback fetch if no cached result
        if (lastEmotionResult == null)
        {
            Debug.Log("[EmotionBridge] No cached result, fetching /latest_emotion as fallback...");
            yield return StartCoroutine(FetchLatestEmotion());
        }

        if (lastEmotionResult == null)
        {
            Debug.Log("[EmotionBridge] No emotion data available.");
            WriteUnifiedRow(null, "no_data");
            yield break;
        }

        string condition = GetCurrentCondition();
        EmotionResult e = lastEmotionResult;

        float emotionSafety      = (float)e.safety;
        float emotionNaturalness = (float)e.naturalness;
        float emotionProgress    = (float)e.progress;

        // Capture original questionnaire values
        float[] origSafety = GetObjectiveValues("Safety");
        float[] origNaturalness = GetObjectiveValues("Naturalness");
        float[] origProgress = GetObjectiveValues("Progress");

        string blendMode;

        if (condition == StudyConditionManager.LLM_ONLY)
        {
            // LLM-only: emotion goes DIRECTLY to BO, replacing questionnaire values
            blendMode = "llm_direct";
            SetObjectiveValues("Safety", emotionSafety);
            SetObjectiveValues("Naturalness", emotionNaturalness);
            SetObjectiveValues("Progress", emotionProgress);

            Debug.Log($"[EmotionBridge] LLM-only — emotion values set DIRECTLY to BO: " +
                      $"S:{emotionSafety:F2} N:{emotionNaturalness:F2} P:{emotionProgress:F2}");
        }
        else // COMBINATION
        {
            // Combination: blend emotion with questionnaire values
            blendMode = "blended";
            BlendObjective("Safety", emotionSafety);
            BlendObjective("Naturalness", emotionNaturalness);
            BlendObjective("Progress", emotionProgress);

            Debug.Log($"[EmotionBridge] Combination — blended emotion into questionnaire values " +
                      $"(weight={emotionWeight})");
        }

        // Capture final values (after direct set or blend)
        float[] finalSafety = GetObjectiveValues("Safety");
        float[] finalNaturalness = GetObjectiveValues("Naturalness");
        float[] finalProgress = GetObjectiveValues("Progress");

        // Log
        LogBlendedRow(e, origSafety, origNaturalness, origProgress,
                      finalSafety, finalNaturalness, finalProgress, blendMode);
        WriteUnifiedRow(e, blendMode, origSafety, origNaturalness, origProgress,
                        finalSafety, finalNaturalness, finalProgress);

        Debug.Log($"[EmotionBridge] Iteration {GetStudyInfo().iteration} applied and logged ({blendMode}).");
        lastEmotionResult = null;
    }

    // ========================================================================
    // OBJECTIVE HELPERS
    // ========================================================================

    /// <summary>
    /// Set ALL sub-measures of an objective to a single value (for llm_only).
    /// </summary>
    private void SetObjectiveValues(string objectiveKey, float value)
    {
        var mgr = GetBoManager();
        if (mgr == null) return;

        foreach (var ob in mgr.objectives)
        {
            if (ob.key == objectiveKey && ob.value.values.Count > 0)
            {
                for (int i = 0; i < ob.value.values.Count; i++)
                {
                    Debug.Log($"[EmotionBridge] {objectiveKey}[{i}]: {ob.value.values[i]:F2} -> {value:F2} (direct)");
                    ob.value.values[i] = value;
                }
                return;
            }
        }
    }

    /// <summary>
    /// Blend a single emotion score with each sub-measure (for combination).
    /// </summary>
    private void BlendObjective(string objectiveKey, float emotionScore)
    {
        var mgr = GetBoManager();
        if (mgr == null) return;

        foreach (var ob in mgr.objectives)
        {
            if (ob.key == objectiveKey && ob.value.values.Count > 0)
            {
                for (int i = 0; i < ob.value.values.Count; i++)
                {
                    float original = ob.value.values[i];
                    float blended = original * (1f - emotionWeight) + emotionScore * emotionWeight;
                    ob.value.values[i] = blended;

                    Debug.Log($"[EmotionBridge] {objectiveKey}[{i}]: {original:F2} -> {blended:F2} " +
                              $"(emotion:{emotionScore:F2}, weight:{emotionWeight})");
                }
                return;
            }
        }
    }

    private float[] GetObjectiveValues(string key)
    {
        var mgr = GetBoManager();
        if (mgr == null) return new float[0];

        foreach (var ob in mgr.objectives)
        {
            if (ob.key == key && ob.value.values.Count > 0)
                return ob.value.values.ToArray();
        }
        return new float[0];
    }

    // ========================================================================
    // LOGGING
    // ========================================================================

    private void LogBlendedRow(EmotionResult e,
        float[] origSafety, float[] origNaturalness, float[] origProgress,
        float[] finalSafety, float[] finalNaturalness, float[] finalProgress,
        string blendMode)
    {
        var info = GetStudyInfo();
        int envIdx = GetCurrentEnvironmentIndex();
        string condition = GetCurrentCondition();
        var ci = CultureInfo.InvariantCulture;

        var parts = new List<string>
        {
            info.userId, info.conditionId, info.groupId,
            info.iteration.ToString(ci), envIdx.ToString(ci), condition,
            DateTime.Now.ToString("O"),
            e.emotion,
            ((float)e.valence).ToString("F4", ci),
            ((float)e.arousal).ToString("F4", ci),
            ((float)e.confidence).ToString("F4", ci),
            ((float)e.safety).ToString("F4", ci),
            ((float)e.naturalness).ToString("F4", ci),
            ((float)e.progress).ToString("F4", ci),
        };

        for (int i = 0; i < 4; i++)
            parts.Add(i < origSafety.Length ? origSafety[i].ToString("F4", ci) : "NULL");
        parts.Add(origNaturalness.Length > 0 ? origNaturalness[0].ToString("F4", ci) : "NULL");
        parts.Add(origProgress.Length > 0 ? origProgress[0].ToString("F4", ci) : "NULL");

        for (int i = 0; i < 4; i++)
            parts.Add(i < finalSafety.Length ? finalSafety[i].ToString("F4", ci) : "NULL");
        parts.Add(finalNaturalness.Length > 0 ? finalNaturalness[0].ToString("F4", ci) : "NULL");
        parts.Add(finalProgress.Length > 0 ? finalProgress[0].ToString("F4", ci) : "NULL");

        parts.Add(emotionWeight.ToString("F2", ci));
        parts.Add(blendMode);

        File.AppendAllText(blendCsvPath, string.Join(",", parts) + "\n");
    }

    private void WriteUnifiedRow(EmotionResult e, string blendMode,
        float[] origSafety = null, float[] origNaturalness = null, float[] origProgress = null,
        float[] finalSafety = null, float[] finalNaturalness = null, float[] finalProgress = null)
    {
        var info = GetStudyInfo();
        int envIdx = GetCurrentEnvironmentIndex();
        string condition = GetCurrentCondition();
        var ci = CultureInfo.InvariantCulture;

        InitUnifiedCsv(info.userId, info.conditionId);

        // If originals not provided, use stored questionnaire values (fresh from WriteResults)
        // Falls back to GetObjectiveValues if no stored values available
        if (origSafety == null)
            origSafety = latestQuestionnaireSafety.Length > 0 ? latestQuestionnaireSafety : GetObjectiveValues("Safety");
        if (origNaturalness == null)
            origNaturalness = latestQuestionnaireNaturalness.Length > 0 ? latestQuestionnaireNaturalness : GetObjectiveValues("Naturalness");
        if (origProgress == null)
            origProgress = latestQuestionnaireProgress.Length > 0 ? latestQuestionnaireProgress : GetObjectiveValues("Progress");

        var parts = new List<string>
        {
            info.userId, info.conditionId, info.groupId,
            info.iteration.ToString(ci), envIdx.ToString(ci), condition,
            DateTime.Now.ToString("O"),
        };

        // Questionnaire Safety1-4
        for (int i = 0; i < 4; i++)
            parts.Add(i < origSafety.Length ? origSafety[i].ToString("F4", ci) : "NULL");
        parts.Add(origNaturalness != null && origNaturalness.Length > 0 ? origNaturalness[0].ToString("F4", ci) : "NULL");
        parts.Add(origProgress != null && origProgress.Length > 0 ? origProgress[0].ToString("F4", ci) : "NULL");

        // Emotion values
        if (e != null)
        {
            parts.Add(e.emotion);
            parts.Add(((float)e.valence).ToString("F4", ci));
            parts.Add(((float)e.arousal).ToString("F4", ci));
            parts.Add(((float)e.confidence).ToString("F4", ci));
            parts.Add(((float)e.safety).ToString("F4", ci));
            parts.Add(((float)e.naturalness).ToString("F4", ci));
            parts.Add(((float)e.progress).ToString("F4", ci));
        }
        else
        {
            parts.AddRange(new[] { "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL" });
        }

        // Final values (what BO actually received)
        if (finalSafety == null) finalSafety = GetObjectiveValues("Safety");
        if (finalNaturalness == null) finalNaturalness = GetObjectiveValues("Naturalness");
        if (finalProgress == null) finalProgress = GetObjectiveValues("Progress");

        for (int i = 0; i < 4; i++)
            parts.Add(i < finalSafety.Length ? finalSafety[i].ToString("F4", ci) : "NULL");
        parts.Add(finalNaturalness.Length > 0 ? finalNaturalness[0].ToString("F4", ci) : "NULL");
        parts.Add(finalProgress.Length > 0 ? finalProgress[0].ToString("F4", ci) : "NULL");

        parts.Add(emotionWeight.ToString("F2", ci));
        parts.Add(blendMode);

        // Current driving parameters
        parts.Add(GetParameterValue("Speed").ToString("F4", ci));
        parts.Add(GetParameterValue("Distance").ToString("F4", ci));
        parts.Add(GetParameterValue("Braking").ToString("F4", ci));

        File.AppendAllText(unifiedCsvPath, string.Join(",", parts) + "\n");
    }

    private IEnumerator FetchLatestEmotion()
    {
        string url = emotionServer + "/latest_emotion";
        UnityWebRequest req = UnityWebRequest.Get(url);
        yield return req.SendWebRequest();

        if (req.result == UnityWebRequest.Result.Success)
        {
            var e = JsonConvert.DeserializeObject<EmotionResult>(req.downloadHandler.text);
            if (e != null && e.confidence > 0.1)
            {
                lastEmotionResult = e;
                LogRawEmotion(e);
            }
        }
        else
        {
            Debug.LogWarning($"[EmotionBridge] /latest_emotion fetch failed: {req.error}");
        }
    }
}

[System.Serializable]
public class EmotionResult
{
    public string emotion;
    public double valence;
    public double arousal;
    public double confidence;
    public double safety;
    public double naturalness;
    public double progress;
}

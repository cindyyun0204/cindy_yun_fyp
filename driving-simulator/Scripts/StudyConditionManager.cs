using UnityEngine;

/// <summary>
/// Determines which study condition is active based on Condition ID.
/// 
/// Conditions:
///   Condition 1 = "llm_only"    — Emotion values go DIRECTLY to BO. Questionnaire is skipped.
///   Condition 2 = "value_only"  — Questionnaire values go DIRECTLY to BO. Emotion system does NOT capture (saves GPU).
///   Condition 3 = "combination" — Both run. Questionnaire values are blended with emotion values using emotionWeight.
///
/// NOTE: conditionId controls the study condition (1/2/3).
///       groupId controls the route order (e.g. "RuralUrbanHighway") — these are separate.
///
/// BO runs in environment 1 only. Final design carries over to environments 2 and 3.
///
/// Usage:
///   string condition = StudyConditionManager.GetCondition(conditionId);
///   bool useEmotion  = StudyConditionManager.ShouldUseEmotion(conditionId);
///   bool useQuestionnaire = StudyConditionManager.ShouldUseQuestionnaireForBO(conditionId);
/// </summary>
public static class StudyConditionManager
{
    public const string LLM_ONLY    = "llm_only";
    public const string VALUE_ONLY  = "value_only";
    public const string COMBINATION = "combination";

    /// <summary>
    /// Get the active condition from the condition ID.
    ///   1 = llm_only
    ///   2 = value_only
    ///   3 = combination
    ///   Default = combination
    /// </summary>
    public static string GetCondition(string conditionId)
    {
        int cond = 0;
        if (!string.IsNullOrEmpty(conditionId))
            int.TryParse(conditionId, out cond);

        switch (cond)
        {
            case 1:  return LLM_ONLY;
            case 2:  return VALUE_ONLY;
            case 3:  return COMBINATION;
            default:
                Debug.LogWarning($"[StudyCondition] Unknown conditionId '{conditionId}', defaulting to combination.");
                return COMBINATION;
        }
    }

    /// <summary>
    /// Should emotion values affect BO?
    /// True for: llm_only (directly), combination (blended)
    /// False for: value_only
    /// </summary>
    public static bool ShouldUseEmotion(string conditionId)
    {
        string c = GetCondition(conditionId);
        return c == LLM_ONLY || c == COMBINATION;
    }

    /// <summary>
    /// Should questionnaire values be sent to BO?
    /// True for: value_only (directly), combination (blended)
    /// False for: llm_only
    /// </summary>
    public static bool ShouldUseQuestionnaireForBO(string conditionId)
    {
        string c = GetCondition(conditionId);
        return c == VALUE_ONLY || c == COMBINATION;
    }

    /// <summary>
    /// Should blending be applied? Only for combination.
    /// </summary>
    public static bool ShouldBlend(string conditionId)
    {
        return GetCondition(conditionId) == COMBINATION;
    }

    /// <summary>
    /// Should the emotion system capture and process data?
    /// True for: llm_only, combination
    /// False for: value_only (saves GPU)
    /// </summary>
    public static bool ShouldEmotionSystemCapture(string conditionId)
    {
        return GetCondition(conditionId) != VALUE_ONLY;
    }

    public static string GetConditionDisplayName(string condition)
    {
        switch (condition)
        {
            case LLM_ONLY:    return "LLM Only (emotion drives BO directly)";
            case VALUE_ONLY:  return "Value Only (questionnaire drives BO directly)";
            case COMBINATION: return "Combination (blended)";
            default:          return "Unknown";
        }
    }
}
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using QuestionnaireToolkit.Scripts;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;
using BOforUnity;

public class LoopForQT : MonoBehaviour
{
    public MeshRenderer window;
    public MeshRenderer doorWindowLeft;
    public MeshRenderer doorWindowRight;
    public float alphaWindow;
    
    public bool timeScaling = false;

    public float currentScale = 0;

    public float targetScale = 0;

    public float t;

    private int currentIteration = 0;

    public bool isTimeFreezed = false;
    
    private bool waitUntilQT = false;
    public float timer = 10f;
    public float timerExtension = 10f;
    public TextMeshProUGUI timerText;
    private float lastWaitStart;

    public GameObject vivePointers;
    public bool inManualScene = false;
    public bool qtShown = false;

    [SerializeField] private GameObject car;
    [HideInInspector] private CarAIController carAIController;
    [HideInInspector] public bool pauseQt = false;

    // Track which environment we're in (0, 1, 2)
    [HideInInspector] public int currentEnvironmentIndex = 0;

    // Study end tracking
    [HideInInspector] public bool studyFinished = false;
    private bool studyFullyEnded = false;
    private float stuckTimer = 0f;
    private Vector3 lastCarPosition;
    private float lastMovementTime;

    [Header("Study End Settings")]
    [Tooltip("Timeout in seconds if car gets stuck (default 300 = 5 minutes)")]
    public float stuckTimeoutSeconds = 300f;
    [Tooltip("Minimum distance the car must move to not be considered stuck")]
    public float stuckDistanceThreshold = 2f;

    public int CURRENT_IT
    {
        get { return currentIteration; }
    }
    public float TIMER
    {
        get { return timer; }
    }

    void Start()
    {
        alphaWindow = 0.3f;
        carAIController = car.GetComponent<CarAIController>();
        lastCarPosition = car.transform.position;
        lastMovementTime = Time.unscaledTime;
    }

    void Update()
    {
        if (timeScaling)
        {
            Time.timeScale = Mathf.Lerp(currentScale, targetScale, t);
            t += 1f * Time.fixedDeltaTime;
        }
        
        if (waitUntilQT)
        {
            timerText.text = (Time.unscaledTime - lastWaitStart).ToString();
        }

        // Check for stuck timeout in first environment only
        if (!studyFinished && currentEnvironmentIndex == 0)
        {
            CheckStuckTimeout();
        }
    }

    /// <summary>
    /// Check if the car is making meaningful progress.
    /// Tracks cumulative distance over a rolling time window.
    /// If distance covered in the last stuckTimeoutSeconds is below threshold, car is "stuck."
    /// This catches both full stops AND inching forward in traffic jams.
    /// </summary>
    private float _stuckCheckLastTime = 0f;
    private float _stuckCheckInterval = 5f; // check every 5 seconds
    private float _cumulativeDistance = 0f;
    private float _windowStartTime = 0f;
    private Vector3 _lastCheckPosition;
    private bool _stuckCheckInitialized = false;

    [Tooltip("Minimum distance the car must cover in the timeout window to not be stuck (default 50 units)")]
    public float stuckMinDistanceInWindow = 50f;

    private void CheckStuckTimeout()
    {
        if (!_stuckCheckInitialized)
        {
            _stuckCheckInitialized = true;
            _lastCheckPosition = car.transform.position;
            _windowStartTime = Time.unscaledTime;
            _stuckCheckLastTime = Time.unscaledTime;
            _cumulativeDistance = 0f;
            return;
        }

        // Only check every few seconds to avoid per-frame overhead
        if (Time.unscaledTime - _stuckCheckLastTime < _stuckCheckInterval)
            return;

        _stuckCheckLastTime = Time.unscaledTime;

        // Add distance since last check
        float distSinceLastCheck = Vector3.Distance(car.transform.position, _lastCheckPosition);
        _cumulativeDistance += distSinceLastCheck;
        _lastCheckPosition = car.transform.position;

        float elapsed = Time.unscaledTime - _windowStartTime;

        // If we've been tracking for the full timeout window
        if (elapsed >= stuckTimeoutSeconds)
        {
            if (_cumulativeDistance < stuckMinDistanceInWindow)
            {
                // Car hasn't made meaningful progress
                if (currentEnvironmentIndex >= 1)
                {
                    Debug.LogWarning($"[LoopForQT] Car only moved {_cumulativeDistance:F1}m in {stuckTimeoutSeconds}s (final env). Ending study.");
                    EndStudy("The study has been completed.\n(Session ended due to traffic conditions.)");
                }
                else
                {
                    Debug.LogWarning($"[LoopForQT] Car only moved {_cumulativeDistance:F1}m in {stuckTimeoutSeconds}s (env {currentEnvironmentIndex}). Forcing transition.");
                    
                    // Reset for next environment
                    ResetStuckCheck();
                    
                    carAIController.updateWaitingForExit();
                    AfterAreaQt();
                }
            }
            else
            {
                // Car is making progress — reset the window
                ResetStuckCheck();
            }
        }
    }

    private void ResetStuckCheck()
    {
        _cumulativeDistance = 0f;
        _windowStartTime = Time.unscaledTime;
        _lastCheckPosition = car.transform.position;
        _stuckCheckInitialized = true;
    }

    public void updateWaitForExit()
    {
        carAIController.updateWaitingForExit();
    }

    public void DarkenWindows()
    {
        var windowColorDoorLeft = doorWindowLeft.material.color;
        var windowColorDoorRight = doorWindowRight.material.color;
        var windowColor = window.material.color;

        window.material.color = new Color(windowColor.r, windowColor.g, windowColor.b, 1);
        doorWindowLeft.material.color = new Color(windowColorDoorLeft.r, windowColorDoorLeft.g, windowColorDoorLeft.b, 1);
        doorWindowRight.material.color = new Color(windowColorDoorRight.r, windowColorDoorRight.g, windowColorDoorRight.b, 1);
    }

    public void ShowWindows()
    {
        var windowColorDoorLeft = doorWindowLeft.material.color;
        var windowColorDoorRight = doorWindowRight.material.color;
        var windowColor = window.material.color;
        window.material.color = new Color(windowColor.r, windowColor.g, windowColor.b, alphaWindow);
        doorWindowLeft.material.color = new Color(windowColorDoorLeft.r, windowColorDoorLeft.g, windowColorDoorLeft.b, alphaWindow);
        doorWindowRight.material.color = new Color(windowColorDoorRight.r, windowColorDoorRight.g, windowColorDoorRight.b, alphaWindow);
    }

    public void freezeScreen()
    {
        timeScaling = true;
        currentScale = Time.timeScale;
        targetScale = 0.025f;
        t = 0f;

        DarkenWindows();

        isTimeFreezed = true;

        qtShown = true;

        WaitForQTCoroutine(0);
    }

    public void unfreezeScreen()
    {
        timeScaling = true;
        currentScale = Time.timeScale;
        targetScale = 1;
        t = 0;

        ShowWindows();
        isTimeFreezed = false;

        qtShown = false;

        WaitForQTCoroutine(1);
    }

    public void AfterAreaQt()
    {
        var seed = carAIController.chosenSeed;
        Portal_Script.Area currentArea = carAIController.currentArea;

        if (!carAIController.routeMap.ContainsKey(seed))
            return;

        Portal_Script.Area[] route = carAIController.routeMap[seed];

        // Check if we've completed all environments
        if (currentEnvironmentIndex >= 1)
        {
            // Second environment done — show final area questionnaire, then end study
            Debug.Log("[LoopForQT] All environments completed. Showing final questionnaire.");

            // Mark as finishing so EndOfQt and other loops don't interfere
            studyFinished = true;

            GameObject qtManager = GameObject.FindGameObjectWithTag("QTManager");
            if (qtManager != null)
            {
                freezeScreen();
                
                // Use questionnaire index 2 (Area 2/3 questionnaire) for the final survey
                var qt = qtManager.GetComponent<QTManager>().questionnaires[2];
                
                // Remove any existing listeners
                qt.onQuestionnaireFinished.RemoveAllListeners();
                
                // Add our end-study listener
                qt.onQuestionnaireFinished.AddListener(() =>
                {
                    Debug.Log("[LoopForQT] Final questionnaire completed. Ending study.");
                    EndStudy("The study has been completed.\nThank you for participating!");
                });
                
                qt.StartQuestionnaire();
            }
            else
            {
                EndStudy("The study has been completed.\nThank you for participating!");
            }
            return;
        }

        int questionnaireIndex = -1;

        if (currentArea == route[0])
        {
            Debug.Log("route[0] öffnet Area1QT");
            questionnaireIndex = 1;
        }
        else if (currentArea == route[1] || currentArea == route[2])
        {
            Debug.Log("route[1/2] öffnet Area2QT");
            questionnaireIndex = 2;
        }

        if (questionnaireIndex != -1)
        {
            currentEnvironmentIndex++;
            Debug.Log($"[LoopForQT] Environment transitioned to index {currentEnvironmentIndex}");

            var bomanager = GameObject.FindGameObjectWithTag("BOforUnityManager")
                ?.GetComponent<BoForUnityManager>();
            if (bomanager != null)
            {
                string condition = StudyConditionManager.GetCondition(bomanager.conditionId);
                Debug.Log($"[LoopForQT] Condition: {StudyConditionManager.GetConditionDisplayName(condition)}");
            }

            // Reset stuck timer for new environment
            lastCarPosition = car.transform.position;
            lastMovementTime = Time.unscaledTime;

            GameObject qtManager = GameObject.FindGameObjectWithTag("QTManager");

            if (qtManager != null)
            {
                freezeScreen();
                updateWaitForExit();
                
                Debug.Log("QTPaused iterati" + pauseQt);
                qtManager.GetComponent<QTManager>().questionnaires[questionnaireIndex].StartQuestionnaire();
                carAIController.updatePopUp(true, "The driving style is now transferred to the next environment and will remain the same. Continue rating when prompted; ratings will not change the driving style.");
            }
            else
            {
                Debug.LogWarning("QTManager mit Tag 'QTManager' nicht gefunden.");
            }
        }
    }

    public void StartQuestionnaireLoop()
    {
        if (studyFinished) return; // Don't start new loops if study is done

        if (!pauseQt)
        {
            StartCoroutine(UserRatingCoroutine(timer));
        }
        else
        {
            unfreezeScreen();
            
            lastWaitStart = Time.unscaledTime;
        }
    }
    
    private IEnumerator WaitForQTCoroutine(int lerpToX)
    {
        Debug.Log("lerp to " + lerpToX);
        yield return new WaitForSecondsRealtime(2);
        timeScaling = false;
        Debug.Log("turn off lerp to " + lerpToX);
    }

    private IEnumerator UserRatingCoroutine(float waitTime)
    {    
        if (studyFinished) yield break; // Safety check

        unfreezeScreen();

        // --- EMOTION: signal iteration START ---
        var emotionBridge = FindObjectOfType<EmotionBOBridge>();
        if (emotionBridge != null)
        {
            emotionBridge.SignalIterationStart();
            Debug.Log("[LoopForQT] Signaled iteration_start to emotion server.");
        }
        // --- END ---

        // Wait for the driving iteration
        waitUntilQT = true;
        lastWaitStart = Time.unscaledTime;
        yield return new WaitForSecondsRealtime(waitTime);
        waitUntilQT = false;

        if (studyFinished) yield break; // Could have ended during wait

        // --- EMOTION: signal iteration END ---
        if (emotionBridge != null)
        {
            emotionBridge.SignalIterationEnd();
            Debug.Log("[LoopForQT] Signaled iteration_end to emotion server.");
        }
        // --- END ---

        currentIteration++;

        // --- Check condition: skip questionnaire for LLM-only ---
        var bomanager = GameObject.FindGameObjectWithTag("BOforUnityManager")
            ?.GetComponent<BoForUnityManager>();
        string condition = bomanager != null 
            ? StudyConditionManager.GetCondition(bomanager.conditionId) 
            : StudyConditionManager.COMBINATION;

        if (condition == StudyConditionManager.LLM_ONLY)
        {
            // LLM-only: skip questionnaire but freeze screen like normal
            freezeScreen();
            
            // Show processing message
            carAIController.updatePopUp(true, "Processing your driving session...\nPlease wait.");
            
            // Wait for LLM to finish processing
            Debug.Log("[LoopForQT] LLM-only — waiting for emotion server to process...");
            yield return new WaitForSecondsRealtime(5f);
            
            // Hide message
            carAIController.updatePopUp(false, "");
            
            Debug.Log("[LoopForQT] LLM-only — auto-submitting 0s to BO.");

            if (bomanager != null)
            {
                // Clear previous iteration's values first
                foreach (var ob in bomanager.objectives)
                {
                    ob.value.values.Clear();
                }

                // Add placeholder 0s
                foreach (var ob in bomanager.objectives)
                {
                    for (int i = 0; i < ob.value.numberOfSubMeasures; i++)
                    {
                        bomanager.optimizer.AddObjectiveValue(ob.key, 0f);
                    }
                }
            }

            LLMOnlyEndOfIteration(bomanager);
        }
        else
        {
            // Value-only or Combination: show questionnaire as normal
            freezeScreen();

            GameObject.FindWithTag("QTManager").GetComponent<QTManager>().questionnaires[0].StartQuestionnaire();
            
            vivePointers.SetActive(true);
        }
    }

    /// <summary>
    /// Handles end-of-iteration for LLM-only condition (no questionnaire shown).
    /// </summary>
    private void LLMOnlyEndOfIteration(BoForUnityManager bomanager)
    {
        if (studyFinished) return;

        // Apply emotion to BO objectives
        var emotionBridge = FindObjectOfType<EmotionBOBridge>();
        if (emotionBridge != null)
        {
            emotionBridge.ApplyEmotionToObjectives();
            Debug.Log("[LoopForQT] LLM-only — emotion applied to objectives.");
        }

        // Run BO optimization
        if (bomanager != null && currentIteration <= carAIController.totalIterationsFixed)
        {
            Debug.Log($"[LoopForQT] LLM-only — iteration {currentIteration}/{carAIController.totalIterationsFixed}");

            if (currentIteration == carAIController.totalIterationsFixed)
            {
                timer += timerExtension;
                bomanager.OptimizationStart();

                Debug.Log("FinalDesign via LLM-only with iteration: " + currentIteration);
                bomanager.SelectAndApplyFinalDesign();
                carAIController.updatePopUp(true, "The System decided on your final design. You will experience this design for the remainder of the environment.");
            }
            else
            {
                bomanager.OptimizationStart();
                Debug.Log("[LoopForQT] LLM-only — BO optimization started.");
            }
        }

        if (currentIteration > carAIController.totalIterationsFixed || bomanager == null)
        {
            Debug.Log("Totalits " + carAIController.totalIterations);

            if (currentIteration == carAIController.totalIterations)
            {
                timer += timerExtension;
            }

            if (currentIteration == carAIController.totalIterations + 1)
            {
                timer -= timerExtension;
                AfterAreaQt();
            }
            else
            {
                Debug.Log("[LoopForQT] LLM-only — starting next iteration.");
                StartQuestionnaireLoop();
            }
        }
    }

    // ========================================================================
    // STUDY END
    // ========================================================================

    /// <summary>
    /// End the study. Freezes everything and shows a completion message with a quit button.
    /// </summary>
    public void EndStudy(string message)
    {
        if (studyFullyEnded) return;
        studyFullyEnded = true;
        studyFinished = true;

        Debug.Log($"[LoopForQT] === STUDY FINISHED === {message}");

        // Stop all coroutines to prevent more iterations
        StopAllCoroutines();

        // Freeze the simulation
        Time.timeScale = 0f;
        timeScaling = false;

        // Close any open questionnaires
        try
        {
            var qtManager = GameObject.FindWithTag("QTManager")?.GetComponent<QTManager>();
            if (qtManager != null)
            {
                foreach (var q in qtManager.questionnaires)
                {
                    if (q.running)
                    {
                        q.HideQuestionnaire();
                        q.running = false;
                    }
                }
            }
        }
        catch { }

        // Close any popups
        try
        {
            carAIController.updatePopUp(false, "");
        }
        catch { }

        // Signal emotion system to stop
        try
        {
            var emotionBridge = FindObjectOfType<EmotionBOBridge>();
            if (emotionBridge != null)
            {
                emotionBridge.SignalIterationEnd();
            }
        }
        catch { }

        // Shut down BO socket
        try
        {
            var bomanager = GameObject.FindGameObjectWithTag("BOforUnityManager")
                ?.GetComponent<BoForUnityManager>();
            if (bomanager != null)
            {
                bomanager.socketNetwork.SocketQuit();
                bomanager.simulationRunning = false;
            }
        }
        catch { }

        // Show end screen
        ShowEndScreen(message);
    }

    /// <summary>
    /// Creates a full-screen end message with a quit button.
    /// </summary>
    private void ShowEndScreen(string message)
    {
        // Create a canvas for the end screen
        GameObject canvasObj = new GameObject("StudyEndCanvas");
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 9999; // On top of everything
        canvasObj.AddComponent<CanvasScaler>();
        canvasObj.AddComponent<GraphicRaycaster>();

        // Dark background
        GameObject bgObj = new GameObject("Background");
        bgObj.transform.SetParent(canvasObj.transform, false);
        Image bg = bgObj.AddComponent<Image>();
        bg.color = new Color(0.1f, 0.1f, 0.1f, 0.95f);
        RectTransform bgRect = bg.GetComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.sizeDelta = Vector2.zero;

        // Message text
        GameObject textObj = new GameObject("MessageText");
        textObj.transform.SetParent(canvasObj.transform, false);
        TextMeshProUGUI text = textObj.AddComponent<TextMeshProUGUI>();
        text.text = message;
        text.fontSize = 36;
        text.alignment = TextAlignmentOptions.Center;
        text.color = Color.white;
        RectTransform textRect = text.GetComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0.1f, 0.5f);
        textRect.anchorMax = new Vector2(0.9f, 0.8f);
        textRect.sizeDelta = Vector2.zero;

        // Quit button
        GameObject buttonObj = new GameObject("QuitButton");
        buttonObj.transform.SetParent(canvasObj.transform, false);
        Image buttonImage = buttonObj.AddComponent<Image>();
        buttonImage.color = new Color(0.2f, 0.6f, 0.2f, 1f);
        Button button = buttonObj.AddComponent<Button>();
        button.targetGraphic = buttonImage;
        RectTransform buttonRect = buttonObj.GetComponent<RectTransform>();
        buttonRect.anchorMin = new Vector2(0.35f, 0.25f);
        buttonRect.anchorMax = new Vector2(0.65f, 0.35f);
        buttonRect.sizeDelta = Vector2.zero;

        // Button text
        GameObject buttonTextObj = new GameObject("ButtonText");
        buttonTextObj.transform.SetParent(buttonObj.transform, false);
        TextMeshProUGUI buttonText = buttonTextObj.AddComponent<TextMeshProUGUI>();
        buttonText.text = "Close Application";
        buttonText.fontSize = 24;
        buttonText.alignment = TextAlignmentOptions.Center;
        buttonText.color = Color.white;
        RectTransform buttonTextRect = buttonText.GetComponent<RectTransform>();
        buttonTextRect.anchorMin = Vector2.zero;
        buttonTextRect.anchorMax = Vector2.one;
        buttonTextRect.sizeDelta = Vector2.zero;

        // Button click handler — need to unpause time briefly for the click to register
        button.onClick.AddListener(() =>
        {
            Debug.Log("[LoopForQT] Quit button pressed. Closing application.");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        });

        // Unpause time so UI can receive clicks
        Time.timeScale = 1f;

        Debug.Log("[LoopForQT] End screen displayed. Waiting for user to quit.");
    }
}

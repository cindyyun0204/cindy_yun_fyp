using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Emotion HUD – attaches to a Canvas and wires up the live webcam feed,
/// latest emotion results, ASR transcript and system status indicators.
///
/// ── How to set up in Unity ───────────────────────────────────────────────
///
///  1. Create a new Canvas (Screen Space – Overlay, CanvasScaler → Scale
///     With Screen Size, ref 1920×1080).
///
///  2. Add an EmotionHUD component to the Canvas root (or any child object).
///
///  3. Use the helper menu item  Tools → Emotion HUD → Build UI  (or just
///     call EmotionHUD.BuildUI() at runtime) to auto-create all child objects,
///     OR wire them manually via the Inspector fields below.
///
///  4. Drag the references into the Inspector slots:
///       runner        →  the GameObject with OfficialMediaPipeRunner
///       emotionClient →  the GameObject with EmotionClient
///       whisperClient →  the GameObject with WhisperXClient  (optional)
///
///  Everything else auto-finds itself on Start().
/// ─────────────────────────────────────────────────────────────────────────
/// </summary>
[AddComponentMenu("Emotion/Emotion HUD")]
public class EmotionHUD : MonoBehaviour
{
    // ── Inspector references ──────────────────────────────────────────────
    [Header("System references (auto-found if left empty)")]
    public OfficialMediaPipeRunner runner;

    [Header("Webcam panel")]
    [Tooltip("RawImage that will display the live webcam feed")]
    public RawImage webcamDisplay;

    [Header("Status bar")]
    public TMP_Text statusLabel;        // bottom one-liner  e.g.  "● Webcam OK  ● Server OK"

    [Header("Colours")]
    public Color colPositive  = new Color(0.20f, 0.80f, 0.40f);   // green (status bar)

    [Header("Refresh")]
    [Tooltip("UI refresh rate (Hz)")]
    public float refreshHz = 10f;

    // ── private state ──────────────────────────────────────────────────────
    private bool   _webcamOk   = false;
    private float  _nextRefresh = 0f;

    // ─────────────────────────────────────────────────────────────────────
    //  Unity lifecycle
    // ─────────────────────────────────────────────────────────────────────

    void Awake()
    {
        if (runner == null) runner = FindObjectOfType<OfficialMediaPipeRunner>();
    }

    void Start()
    {
        // Subscribe to ASR updates so the transcript box stays live
        // (we poll WhisperXClient.latestText in Update instead, so no event needed)
        ClearAll();
    }

    void Update()
    {
        float now = Time.unscaledTime;
        if (now < _nextRefresh) return;
        _nextRefresh = now + 1f / Mathf.Max(1f, refreshHz);

        RefreshWebcam();
        RefreshStatus();
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Refresh helpers
    // ─────────────────────────────────────────────────────────────────────

    void RefreshWebcam()
    {
        if (webcamDisplay == null) return;
        if (runner == null) return;

        WebCamTexture wcTex = runner.WebcamTexture;

        if (wcTex != null && wcTex.isPlaying && wcTex.width > 16)
        {
            if (webcamDisplay.texture != wcTex)
                webcamDisplay.texture = wcTex;

            // Mirror correction: WebCamTexture can come in upside-down on some
            // platforms.  videoVerticallyMirrored == true means we need to flip.
            float scaleY = wcTex.videoVerticallyMirrored ? -1f : 1f;
            webcamDisplay.uvRect = new Rect(0f, scaleY < 0 ? 1f : 0f, 1f, scaleY);

            _webcamOk = true;
        }
        else
        {
            _webcamOk = false;
        }
    }

    void RefreshStatus()
    {
        if (statusLabel == null) return;
        string wcStatus = _webcamOk ? "<color=#33CC66>● Webcam</color>" : "<color=#CC4444>● Webcam</color>";
        statusLabel.text = wcStatus;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Utilities
    // ─────────────────────────────────────────────────────────────────────

    void ClearAll()
    {
        SetSafe(statusLabel, "Waiting for data…");
    }

    static void SetSafe(TMP_Text t, string s) { if (t != null) t.text = s; }

    // ─────────────────────────────────────────────────────────────────────
    //  Static builder – call from Editor menu or at runtime
    // ─────────────────────────────────────────────────────────────────────

#if UNITY_EDITOR
    [UnityEditor.MenuItem("Tools/Emotion HUD/Build UI on selected Canvas")]
    static void MenuBuildUI()
    {
        var go = UnityEditor.Selection.activeGameObject;
        if (go == null) { Debug.LogWarning("Select the Canvas first."); return; }
        var canvas = go.GetComponent<Canvas>();
        if (canvas == null) { Debug.LogWarning("Selected object has no Canvas."); return; }

        var hud = go.GetComponent<EmotionHUD>() ?? go.AddComponent<EmotionHUD>();
        hud.BuildUI(canvas);
        UnityEditor.EditorUtility.SetDirty(go);
        Debug.Log("[EmotionHUD] UI built successfully.");
    }
#endif

    /// <summary>
    /// Procedurally creates the full HUD hierarchy under <paramref name="canvas"/>.
    /// Call this from an Editor menu item or your own bootstrap code.
    /// All existing child objects named "EmotionHUD_*" are destroyed first.
    /// </summary>
    public void BuildUI(Canvas canvas)
    {
        if (canvas == null) return;
        Transform root = canvas.transform;

        // Clear previous auto-build
        for (int i = root.childCount - 1; i >= 0; i--)
        {
            var c = root.GetChild(i);
            if (c.name.StartsWith("EmotionHUD_"))
            {
#if UNITY_EDITOR
                DestroyImmediate(c.gameObject);
#else
                Destroy(c.gameObject);
#endif
            }
        }

        // ── Webcam panel (full canvas minus status bar) ───────────────────
        var webcamPanel = MakePanel(root, "EmotionHUD_WebcamPanel",
            new Vector2(0f, 0.07f),
            new Vector2(1f, 1.0f),
            new Vector2(5f,  5f),
            new Vector2(-5f, -5f));

        webcamDisplay = MakeRawImage(webcamPanel, "EmotionHUD_WebcamDisplay",
            Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);

        var camLabel = MakeText(webcamPanel, "EmotionHUD_CamLabel",
            new Vector2(0f, 0f), new Vector2(0.1f, 0.05f),
            Vector2.zero, Vector2.zero,
            "LIVE", 16, TextAlignmentOptions.BottomLeft);
        camLabel.color = new Color(1, 1, 1, 0.55f);

        // ── Status bar (bottom strip) ─────────────────────────────────────
        var statusPanel = MakePanel(root, "EmotionHUD_StatusPanel",
            new Vector2(0f,    0.0f),
            new Vector2(1.0f,  0.07f),
            new Vector2(0f,    0f),
            new Vector2(0f,    0f));

        var bg = statusPanel.GetComponent<Image>();
        if (bg != null) bg.color = new Color(0f, 0f, 0f, 0.8f);

        statusLabel = MakeText(statusPanel, "EmotionHUD_StatusLabel",
            new Vector2(0f, 0f), new Vector2(1f, 1f),
            new Vector2(10f, 0f), new Vector2(-10f, 0f),
            "Waiting for data…", 14, TextAlignmentOptions.MidlineLeft);
        statusLabel.richText = true;

        Debug.Log("[EmotionHUD] BuildUI complete.");
    }

    // ─────────────────────────────────────────────────────────────────────
    //  UI factory helpers
    // ─────────────────────────────────────────────────────────────────────

    static RectTransform MakeRT(Transform parent, string name)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin    = Vector2.zero;
        rt.anchorMax    = Vector2.one;
        rt.offsetMin    = Vector2.zero;
        rt.offsetMax    = Vector2.zero;
        return rt;
    }

    static RectTransform SetAnchors(RectTransform rt,
        Vector2 aMin, Vector2 aMax, Vector2 offMin, Vector2 offMax)
    {
        rt.anchorMin = aMin; rt.anchorMax = aMax;
        rt.offsetMin = offMin; rt.offsetMax = offMax;
        return rt;
    }

    static Transform MakePanel(Transform parent, string name,
        Vector2 aMin, Vector2 aMax, Vector2 offMin, Vector2 offMax)
    {
        var rt = MakeRT(parent, name);
        SetAnchors(rt, aMin, aMax, offMin, offMax);
        var img = rt.gameObject.AddComponent<Image>();
        img.color = new Color(0.05f, 0.05f, 0.08f, 0.85f);
        return rt.transform;
    }

    static RawImage MakeRawImage(Transform parent, string name,
        Vector2 aMin, Vector2 aMax, Vector2 offMin, Vector2 offMax)
    {
        var rt = MakeRT(parent, name);
        SetAnchors(rt, aMin, aMax, offMin, offMax);
        return rt.gameObject.AddComponent<RawImage>();
    }

    static TMP_Text MakeText(Transform parent, string name,
        Vector2 aMin, Vector2 aMax, Vector2 offMin, Vector2 offMax,
        string defaultText, int fontSize, TextAlignmentOptions align)
    {
        var rt = MakeRT(parent, name);
        SetAnchors(rt, aMin, aMax, offMin, offMax);
        var tmp = rt.gameObject.AddComponent<TextMeshProUGUI>();
        tmp.text      = defaultText;
        tmp.fontSize  = fontSize;
        tmp.alignment = align;
        tmp.color     = Color.white;
        tmp.enableWordWrapping = false;
        tmp.overflowMode = TextOverflowModes.Ellipsis;
        return tmp;
    }

}
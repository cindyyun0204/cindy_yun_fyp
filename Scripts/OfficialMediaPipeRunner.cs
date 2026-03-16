using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OfficialMediaPipeRunner : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private OfficialMediaPipeClient mediaPipeClient;
    [SerializeField] private FaceLandmarksJsonlLogger logger;
    [SerializeField] private EmotionClient emotionClient;
    [SerializeField] private EmotionWindowAggregator aggregator;

    [Header("Enable / Disable")]
    [SerializeField] private bool enableFacePipeline = true;

    [Header("Webcam")]
    [SerializeField] private string webcamDeviceName = "";
    [SerializeField] private int requestedWidth = 640;
    [SerializeField] private int requestedHeight = 480;
    [SerializeField] private int requestedFps = 30;
    [SerializeField] private bool preferFrontFacing = true;

    [Header("Capture")]
    [Tooltip("Times per second. For every 3 seconds set 0.333")]
    [SerializeField] private float captureHz = 0.333f;

    [Header("Face Crop")]
    [SerializeField] private bool enableFaceCrop = true;
    [SerializeField] private float cropPaddingPct = 0.15f;
    [SerializeField] private bool ovalTransparentOutside = true;
    [SerializeField] private bool flipOutputVertically = true;

    [Header("Encode")]
    [SerializeField] private bool sendAsPng = false;
    [SerializeField, Range(1, 100)] private int jpgQuality = 80;

    [Header("Debug")]
    [SerializeField] private bool logNoFace = false;

    // ── public read-only access so EmotionHUD can display the live feed ──
    public WebCamTexture WebcamTexture => _webcam;

    private float _nextCaptureTime = 0f;
    private bool _requestInFlight = false;

    private WebCamTexture _webcam;

    // We keep ONE Texture2D whose pixel buffer we fill each capture.
    // It is created with TextureFormat.RGBA32 and is ALWAYS CPU-readable
    // (we never call Apply with makeNoLongerReadable=true).
    private Texture2D _frameTex;
    private Color32[] _webcamPixels;
    private Texture2D _lastCrop;

    void Awake()
    {
        if (emotionClient != null && aggregator == null)
            aggregator = emotionClient.GetComponent<EmotionWindowAggregator>();

        if (emotionClient == null)
            emotionClient = FindObjectOfType<EmotionClient>();

        if (aggregator == null)
            aggregator = FindObjectOfType<EmotionWindowAggregator>();

        if (mediaPipeClient == null)
            mediaPipeClient = GetComponent<OfficialMediaPipeClient>();

        if (mediaPipeClient == null)
            mediaPipeClient = FindObjectOfType<OfficialMediaPipeClient>();
    }

    void Start()
    {
        if (!enableFacePipeline) return;
        StartWebcam();
    }

    void OnDestroy()
    {
        if (_webcam != null && _webcam.isPlaying)
            _webcam.Stop();

        if (_frameTex != null) Destroy(_frameTex);
        if (_lastCrop  != null) Destroy(_lastCrop);
    }

    void Update()
    {
        if (!enableFacePipeline) return;
        if (_requestInFlight) return;
        if (_webcam == null || !_webcam.isPlaying || _webcam.width <= 16 || _webcam.height <= 16) return;
        if (!_webcam.didUpdateThisFrame) return;
        if (mediaPipeClient == null) return;

        float now = Time.unscaledTime;
        if (now < _nextCaptureTime) return;
        _nextCaptureTime = now + (1f / Mathf.Max(0.001f, captureHz));

        StartCoroutine(CaptureAndProcess());
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Webcam startup
    // ─────────────────────────────────────────────────────────────────────────

    private void StartWebcam()
    {
        if (WebCamTexture.devices == null || WebCamTexture.devices.Length == 0)
        {
            Debug.LogError("[OfficialMediaPipeRunner] No webcam devices found.");
            return;
        }

        string deviceToUse = webcamDeviceName;
        if (string.IsNullOrWhiteSpace(deviceToUse))
        {
            for (int i = 0; i < WebCamTexture.devices.Length; i++)
            {
                if (WebCamTexture.devices[i].isFrontFacing == preferFrontFacing)
                {
                    deviceToUse = WebCamTexture.devices[i].name;
                    break;
                }
            }

            if (string.IsNullOrWhiteSpace(deviceToUse))
                deviceToUse = WebCamTexture.devices[0].name;
        }

        _webcam = new WebCamTexture(deviceToUse, requestedWidth, requestedHeight, requestedFps);
        _webcam.Play();

        Debug.Log($"[OfficialMediaPipeRunner] Webcam started: {deviceToUse}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Main capture + process coroutine
    // ─────────────────────────────────────────────────────────────────────────

    private IEnumerator CaptureAndProcess()
    {
        _requestInFlight = true;

        // ── 1. Snapshot webcam pixels into a CPU-readable Texture2D ──────────
        //
        // FIX FOR BLACK IMAGES:
        //   _frameTex is kept CPU-readable at all times (Apply never marks it
        //   non-readable).  We write the pixel buffer ourselves via SetPixels32
        //   so GetPixels32() inside FaceCropper always returns valid data.
        //
        Texture2D frame = SnapshotWebcam();
        if (frame == null)
        {
            _requestInFlight = false;
            yield break;
        }

        // ── 2. Encode a COPY for the HTTP request ────────────────────────────
        //   We must NOT encode _frameTex itself here and then overwrite it next
        //   frame before the coroutine finishes — make an explicit copy.
        byte[] imageBytes = EncodeTexture(frame);

        // ── 3. Send to MediaPipe service ─────────────────────────────────────
        OfficialMediaPipeResponse resp = null;
        yield return StartCoroutine(mediaPipeClient.DetectFromImageBytes(imageBytes, r => resp = r));

        if (resp == null || !resp.ok || !resp.face_detected ||
            resp.landmarks == null || resp.landmarks.Length == 0)
        {
            if (logNoFace)
                Debug.Log("[OfficialMediaPipeRunner] No face detected.");
            _requestInFlight = false;
            yield break;
        }

        Vector3[] lm = ConvertLandmarks(resp.landmarks);
        if (lm == null || lm.Length < 468)
        {
            _requestInFlight = false;
            yield break;
        }

        string utcIso = DateTime.UtcNow.ToString("o");

        // ── 4. Log landmarks ─────────────────────────────────────────────────
        if (logger != null)
            logger.LogLandmarks(lm, faceIndex: 0, utcIso: utcIso);

        // ── 5. Extract features ───────────────────────────────────────────────
        FaceFeatures feats = FaceFeatureExtractor.From468(lm);

        if (aggregator != null)
            aggregator.AddFaceFeatures(feats);

        if (emotionClient != null)
        {
            emotionClient.SetLatestFaceFeatures(utcIso, feats);

            // Forward blendshapes from MediaPipe to EmotionClient
            if (resp.blendshapes != null && resp.blendshapes.Count > 0)
                emotionClient.SetLatestBlendshapes(resp.blendshapes);
        }

        // ── 6. Face crop & save ───────────────────────────────────────────────
        if (emotionClient != null && enableFaceCrop)
        {
            // Pass _webcamPixels (raw CPU buffer) directly — bypasses
            // Texture2D.GetPixels32() which was returning grey pixels.
            Texture2D crop = FaceCropper.CropFace(
                _webcamPixels,
                _webcam.width,
                _webcam.height,
                lm,
                paddingPct: cropPaddingPct
            );

            if (crop != null)
            {
                byte[] cropBytes = EncodeTextureDirect(crop);
                string mime = sendAsPng ? "image/png" : "image/jpeg";

                if (_lastCrop != null) Destroy(_lastCrop);
                _lastCrop = crop;

                emotionClient.SetLatestFaceImage(cropBytes, mime, utcIso);
            }
        }

        _requestInFlight = false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Reads the latest WebCamTexture frame into a CPU-readable Texture2D.
    /// The texture is reused across frames; its pixel data is always valid on
    /// the CPU side because we use SetPixels32 + Apply(false, false) — never
    /// Apply(true, true) which would upload-only and discard the CPU copy.
    /// </summary>
    private Texture2D SnapshotWebcam()
    {
        if (_webcam == null || _webcam.width <= 16 || _webcam.height <= 16)
            return null;

        int w = _webcam.width;
        int h = _webcam.height;

        // Recreate only when dimensions change
        if (_frameTex == null || _frameTex.width != w || _frameTex.height != h)
        {
            if (_frameTex != null) Destroy(_frameTex);
            // isReadable = true is the default when you create via new Texture2D
            _frameTex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        }

        if (_webcamPixels == null || _webcamPixels.Length != w * h)
            _webcamPixels = new Color32[w * h];

        _webcam.GetPixels32(_webcamPixels);

        _frameTex.SetPixels32(_webcamPixels);
        // Apply(updateMipmaps: false, makeNoLongerReadable: false)
        // → uploads to GPU but KEEPS the CPU copy so GetPixels32 still works
        _frameTex.Apply(false, false);

        return _frameTex;
    }

    /// <summary>
    /// Encodes the shared _frameTex by first copying pixels into a fresh
    /// temporary texture so the encode is not racing with the next capture.
    /// </summary>
    private byte[] EncodeTexture(Texture2D src)
    {
        // For the HTTP send we need a stable snapshot; copy pixels first.
        int w = src.width;
        int h = src.height;
        var tmp = new Texture2D(w, h, TextureFormat.RGBA32, false);
        tmp.SetPixels32(src.GetPixels32());
        tmp.Apply(false, false);
        byte[] bytes = sendAsPng ? tmp.EncodeToPNG() : tmp.EncodeToJPG(jpgQuality);
        Destroy(tmp);
        return bytes;
    }

    /// <summary>Encodes a dedicated texture directly (already a private copy).</summary>
    private byte[] EncodeTextureDirect(Texture2D tex)
    {
        return sendAsPng ? tex.EncodeToPNG() : tex.EncodeToJPG(jpgQuality);
    }

    private static Vector3[] ConvertLandmarks(OfficialMediaPipeLandmark[] pts)
    {
        if (pts == null || pts.Length == 0) return Array.Empty<Vector3>();

        int maxIndex = -1;
        for (int i = 0; i < pts.Length; i++)
            if (pts[i] != null && pts[i].i > maxIndex)
                maxIndex = pts[i].i;

        if (maxIndex < 0) return Array.Empty<Vector3>();

        Vector3[] arr = new Vector3[maxIndex + 1];
        for (int i = 0; i < pts.Length; i++)
        {
            var p = pts[i];
            if (p == null) continue;
            if (p.i < 0 || p.i >= arr.Length) continue;
            arr[p.i] = new Vector3(p.x, p.y, p.z);
        }
        return arr;
    }

    private static void FlipTextureVerticalInPlace(Texture2D tex)
    {
        int w = tex.width;
        int h = tex.height;
        Color32[] pixels = tex.GetPixels32();

        for (int y = 0; y < h / 2; y++)
        {
            int yOpp = h - 1 - y;
            int rowA = y * w;
            int rowB = yOpp * w;

            for (int x = 0; x < w; x++)
            {
                int iA = rowA + x;
                int iB = rowB + x;
                (pixels[iA], pixels[iB]) = (pixels[iB], pixels[iA]);
            }
        }

        tex.SetPixels32(pixels);
        tex.Apply(false, false);
    }
}
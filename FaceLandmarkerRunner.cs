// Copyright (c) 2023 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using Mediapipe.Tasks.Vision.FaceLandmarker;
using UnityEngine;
using UnityEngine.Rendering;

namespace Mediapipe.Unity.Sample.FaceLandmarkDetection
{
  public class FaceLandmarkerRunner : VisionTaskApiRunner<FaceLandmarker>
  {
    [SerializeField] private FaceLandmarkerResultAnnotationController _faceLandmarkerResultAnnotationController;
    [SerializeField] private FaceLandmarksJsonlLogger logger;

    [Header("EmotionSystem (has EmotionClient + EmotionWindowAggregator components)")]
    [SerializeField] private EmotionClient emotionClient;
    [SerializeField] private EmotionWindowAggregator aggregator;

    [Header("Face Crop / Capture")]
    [SerializeField] private bool enableFaceCrop = true;

    [Tooltip("Times per second. For every 3 seconds set 0.333")]
    [SerializeField] private float captureHz = 0.333f;

    [SerializeField] private float cropPaddingPct = 0.15f;
    [SerializeField] private bool ovalTransparentOutside = true;
    [SerializeField] private bool flipOutputVertically = true;

    [Header("Encode")]
    [SerializeField] private bool sendAsPng = false;
    [SerializeField, Range(1, 100)] private int jpgQuality = 80;

    private Experimental.TextureFramePool _textureFramePool;
    public readonly FaceLandmarkDetectionConfig config = new FaceLandmarkDetectionConfig();

    private bool _flipH;
    private bool _flipV;
    private int _rotationDeg;

    private readonly object _lock = new object();
    private FaceLandmarkerResult _pendingResult;
    private bool _hasPendingResult;

    private float _nextCaptureTime = 0f;

    private Texture2D _cpuFrame;
    private Texture2D _rotFrame90;
    private Texture2D _rotFrame180;
    private Texture2D _rotFrame270;

    private Texture2D _lastCrop;

    private void Awake()
    {
      // Auto-wire if user only drags EmotionClient
      if (emotionClient != null && aggregator == null)
      {
        aggregator = emotionClient.GetComponent<EmotionWindowAggregator>();
      }
    }

    public override void Stop()
    {
      base.Stop();
      _textureFramePool?.Dispose();
      _textureFramePool = null;

      if (_lastCrop != null)
      {
        Destroy(_lastCrop);
        _lastCrop = null;
      }
    }

    protected override IEnumerator Run()
    {
      yield return AssetLoader.PrepareAssetAsync(config.ModelPath);

      var options = config.GetFaceLandmarkerOptions(
        config.RunningMode == Tasks.Vision.Core.RunningMode.LIVE_STREAM ? OnFaceLandmarkDetectionOutput : null
      );

      taskApi = FaceLandmarker.CreateFromOptions(options, GpuManager.GpuResources);

      var imageSource = ImageSourceProvider.ImageSource;
      yield return imageSource.Play();

      if (!imageSource.isPrepared)
      {
        Debug.LogError("Failed to start ImageSource, exiting...");
        yield break;
      }

      _textureFramePool = new Experimental.TextureFramePool(
        imageSource.textureWidth, imageSource.textureHeight, TextureFormat.RGBA32, 10
      );

      screen.Initialize(imageSource);
      SetupAnnotationController(_faceLandmarkerResultAnnotationController, imageSource);

      var transformationOptions = imageSource.GetTransformationOptions();
      _flipH = transformationOptions.flipHorizontally;
      _flipV = transformationOptions.flipVertically;
      _rotationDeg = (int)transformationOptions.rotationAngle;

      var imageProcessingOptions = new Tasks.Vision.Core.ImageProcessingOptions(
        rotationDegrees: _rotationDeg
      );

      AsyncGPUReadbackRequest req = default;
      var waitUntilReqDone = new WaitUntil(() => req.done);
      var waitForEndOfFrame = new WaitForEndOfFrame();
      var result = FaceLandmarkerResult.Alloc(options.numFaces);

      var canUseGpuImage = SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3 && GpuManager.GpuResources != null;
      using var glContext = canUseGpuImage ? GpuManager.GetGlContext() : null;

      while (true)
      {
        if (isPaused)
          yield return new WaitWhile(() => isPaused);

        if (!_textureFramePool.TryGetTextureFrame(out var textureFrame))
        {
          yield return null;
          continue;
        }

        Image image;
        switch (config.ImageReadMode)
        {
          case ImageReadMode.GPU:
            if (!canUseGpuImage) throw new Exception("ImageReadMode.GPU is not supported");
            textureFrame.ReadTextureOnGPU(imageSource.GetCurrentTexture(), _flipH, _flipV);
            image = textureFrame.BuildGPUImage(glContext);
            yield return waitForEndOfFrame;
            break;

          case ImageReadMode.CPU:
            yield return waitForEndOfFrame;
            textureFrame.ReadTextureOnCPU(imageSource.GetCurrentTexture(), _flipH, _flipV);
            image = textureFrame.BuildCPUImage();
            textureFrame.Release();
            break;

          case ImageReadMode.CPUAsync:
          default:
            req = textureFrame.ReadTextureAsync(imageSource.GetCurrentTexture(), _flipH, _flipV);
            yield return waitUntilReqDone;

            if (req.hasError)
            {
              Debug.LogWarning("Failed to read texture from the image source");
              continue;
            }
            image = textureFrame.BuildCPUImage();
            textureFrame.Release();
            break;
        }

        switch (taskApi.runningMode)
        {
          case Tasks.Vision.Core.RunningMode.IMAGE:
            if (taskApi.TryDetect(image, imageProcessingOptions, ref result))
              _faceLandmarkerResultAnnotationController.DrawNow(result);
            else
              _faceLandmarkerResultAnnotationController.DrawNow(default);
            break;

          case Tasks.Vision.Core.RunningMode.VIDEO:
            if (taskApi.TryDetectForVideo(image, GetCurrentTimestampMillisec(), imageProcessingOptions, ref result))
              _faceLandmarkerResultAnnotationController.DrawNow(result);
            else
              _faceLandmarkerResultAnnotationController.DrawNow(default);
            break;

          case Tasks.Vision.Core.RunningMode.LIVE_STREAM:
            taskApi.DetectAsync(image, GetCurrentTimestampMillisec(), imageProcessingOptions);
            break;
        }
      }
    }

    private void OnFaceLandmarkDetectionOutput(FaceLandmarkerResult result, Image image, long timestamp)
    {
      _faceLandmarkerResultAnnotationController.DrawLater(result);

      var clone = FaceLandmarkerResult.Alloc(1);
      result.CloneTo(ref clone);

      lock (_lock)
      {
        TryDispose(_pendingResult);
        _pendingResult = clone;
        _hasPendingResult = true;
      }
    }

    private void Update()
    {
      if (!_hasPendingResult) return;

      float now = Time.unscaledTime;

      if (enableFaceCrop)
      {
        if (now < _nextCaptureTime) return;
        _nextCaptureTime = now + (1f / Mathf.Max(0.001f, captureHz));
      }

      FaceLandmarkerResult r = default;
      bool got = false;

      lock (_lock)
      {
        if (_hasPendingResult)
        {
          r = _pendingResult;
          _pendingResult = default;
          _hasPendingResult = false;
          got = true;
        }
      }

      if (!got) return;

      var lm = BuildIndexedLandmarksViaReflection(r);
      if (lm == null || lm.Length == 0)
      {
        TryDispose(r);
        return;
      }

      string utcIso = DateTime.UtcNow.ToString("o");

      // 1) log raw landmarks
      if (logger != null)
        logger.LogLandmarks(lm, faceIndex: 0, utcIso: utcIso);

      // 2) compute features
      FaceFeatures feats = FaceFeatureExtractor.From468(lm);

      // 3) feed aggregator window
      if (aggregator != null)
        aggregator.AddFaceFeatures(feats);

      // 4) store latest in EmotionClient too
      if (emotionClient != null)
        emotionClient.SetLatestFaceFeatures(utcIso, feats);

      // 5) crop image -> EmotionClient
      if (emotionClient != null && enableFaceCrop)
      {
        var tex = ImageSourceProvider.ImageSource?.GetCurrentTexture();
        var frame = CopyCurrentFrameToTextureMatchingMediapipe(tex);
        if (frame != null)
        {
          Texture2D crop = FaceCropper.CropFace(
            frame,
            lm,
            transparentOutside: ovalTransparentOutside,
            paddingPct: cropPaddingPct
          );

          if (crop != null)
          {
            if (flipOutputVertically)
              FlipTextureVerticalInPlace(crop);

            byte[] bytes = sendAsPng ? crop.EncodeToPNG() : crop.EncodeToJPG(jpgQuality);
            string mime = sendAsPng ? "image/png" : "image/jpeg";

            if (_lastCrop != null) Destroy(_lastCrop);
            _lastCrop = crop;

            emotionClient.SetLatestFaceImage(bytes, mime, utcIso);
          }
        }
      }

      TryDispose(r);
    }

    private Texture2D CopyCurrentFrameToTextureMatchingMediapipe(Texture srcTex)
    {
      if (srcTex == null) return null;

      int w = srcTex.width;
      int h = srcTex.height;

      if (_cpuFrame == null || _cpuFrame.width != w || _cpuFrame.height != h)
        _cpuFrame = new Texture2D(w, h, TextureFormat.RGBA32, false);

      var prev = RenderTexture.active;
      var rt = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32);

      var scale = new Vector2(_flipH ? -1f : 1f, _flipV ? -1f : 1f);
      var offset = new Vector2(_flipH ? 1f : 0f, _flipV ? 1f : 0f);
      Graphics.Blit(srcTex, rt, scale, offset);

      RenderTexture.active = rt;
      _cpuFrame.ReadPixels(new UnityEngine.Rect(0, 0, w, h), 0, 0);
      _cpuFrame.Apply(false, false);

      RenderTexture.active = prev;
      RenderTexture.ReleaseTemporary(rt);

      if (_rotationDeg == 90)  return Rotate90CW(_cpuFrame, ref _rotFrame90);
      if (_rotationDeg == 180) return Rotate180(_cpuFrame, ref _rotFrame180);
      if (_rotationDeg == 270) return Rotate90CCW(_cpuFrame, ref _rotFrame270);

      return _cpuFrame;
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
          var tmp = pixels[iA];
          pixels[iA] = pixels[iB];
          pixels[iB] = tmp;
        }
      }

      tex.SetPixels32(pixels);
      tex.Apply(false, false);
    }

    private static Texture2D EnsureTex(ref Texture2D t, int w, int h, TextureFormat fmt)
    {
      if (t == null || t.width != w || t.height != h || t.format != fmt)
        t = new Texture2D(w, h, fmt, false);
      return t;
    }

    private static Texture2D Rotate90CW(Texture2D src, ref Texture2D dst)
    {
      int w = src.width, h = src.height;
      dst = EnsureTex(ref dst, h, w, src.format);

      var sp = src.GetPixels32();
      var dp = dst.GetPixels32();

      for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        int newX = h - 1 - y;
        int newY = x;
        dp[newY * h + newX] = sp[y * w + x];
      }

      dst.SetPixels32(dp);
      dst.Apply(false, false);
      return dst;
    }

    private static Texture2D Rotate90CCW(Texture2D src, ref Texture2D dst)
    {
      int w = src.width, h = src.height;
      dst = EnsureTex(ref dst, h, w, src.format);

      var sp = src.GetPixels32();
      var dp = dst.GetPixels32();

      for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        int newX = y;
        int newY = (w - 1 - x);
        dp[newY * h + newX] = sp[y * w + x];
      }

      dst.SetPixels32(dp);
      dst.Apply(false, false);
      return dst;
    }

    private static Texture2D Rotate180(Texture2D src, ref Texture2D dst)
    {
      int w = src.width, h = src.height;
      dst = EnsureTex(ref dst, w, h, src.format);

      var sp = src.GetPixels32();
      var dp = dst.GetPixels32();

      int n = sp.Length;
      for (int i = 0; i < n; i++)
        dp[n - 1 - i] = sp[i];

      dst.SetPixels32(dp);
      dst.Apply(false, false);
      return dst;
    }

    private static Vector3[] BuildIndexedLandmarksViaReflection(FaceLandmarkerResult result)
    {
      object facesObj = result.faceLandmarks;
      if (facesObj == null) return null;

      IEnumerable facesEnum = AsEnumerableOrSingle(facesObj);

      foreach (var faceObj in facesEnum)
      {
        if (faceObj == null) continue;

        object landmarksObj =
            TryGetMemberOrMethod(faceObj, "landmarks")
         ?? TryGetMemberOrMethod(faceObj, "Landmarks")
         ?? TryGetMemberOrMethod(faceObj, "landmark")
         ?? TryGetMemberOrMethod(faceObj, "Landmark");

        if (landmarksObj == null) continue;

        IEnumerable lmEnum = AsEnumerableOrSingle(landmarksObj);

        var list = new List<Vector3>(478);
        foreach (var lm in lmEnum)
        {
          if (lm == null) continue;

          double x = GetDouble(lm, "x", "X");
          double y = GetDouble(lm, "y", "Y");
          double z = GetDouble(lm, "z", "Z");

          if (double.IsNaN(x) || double.IsNaN(y)) continue;
          list.Add(new Vector3((float)x, (float)y, (float)z));
        }

        if (list.Count > 0) return list.ToArray();
      }

      return null;
    }

    private static object TryGetMemberOrMethod(object obj, string name)
    {
      if (obj == null) return null;
      var t = obj.GetType();

      var p = t.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
      if (p != null) return p.GetValue(obj);

      var f = t.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
      if (f != null) return f.GetValue(obj);

      var m = t.GetMethod(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null, Type.EmptyTypes, null);
      if (m != null) return m.Invoke(obj, null);

      return null;
    }

    private static IEnumerable AsEnumerableOrSingle(object obj)
    {
      if (obj == null) return Array.Empty<object>();
      if (obj is string) return new object[] { obj };
      if (obj is IEnumerable e) return e;
      return new object[] { obj };
    }

    private static double GetDouble(object obj, params string[] names)
    {
      var t = obj.GetType();
      foreach (var name in names)
      {
        var p = t.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (p != null)
        {
          var v = p.GetValue(obj);
          if (v != null) return Convert.ToDouble(v);
        }

        var f = t.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (f != null)
        {
          var v = f.GetValue(obj);
          if (v != null) return Convert.ToDouble(v);
        }
      }
      return double.NaN;
    }

    private static void TryDispose(object maybeDisposable)
    {
      if (maybeDisposable is IDisposable d)
      {
        try { d.Dispose(); } catch { }
        return;
      }

      try
      {
        var m = maybeDisposable.GetType().GetMethod("Dispose", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        if (m != null && m.GetParameters().Length == 0)
          m.Invoke(maybeDisposable, null);
      }
      catch { }
    }
  }
}

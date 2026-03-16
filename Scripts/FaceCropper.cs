using UnityEngine;

public static class FaceCropper
{
    /// <summary>
    /// Crops the face region from a raw pixel buffer.
    /// Pass _webcamPixels directly so we never rely on Texture2D.GetPixels32(),
    /// which can return grey/black when the texture was uploaded GPU-only.
    /// </summary>
    public static Texture2D CropFace(
        Color32[] srcPixels,
        int srcWidth,
        int srcHeight,
        Vector3[] landmarks,
        float paddingPct = 0.15f
    )
    {
        if (srcPixels == null || srcPixels.Length == 0) return null;
        if (landmarks == null || landmarks.Length == 0) return null;

        // ── Find landmark bounding box (normalised 0..1) ──────────────────
        // MediaPipe Y=0 is image TOP, but WebCamTexture pixel buffer Y=0 is
        // BOTTOM. Flip Y here so the crop rect lines up with actual pixels.
        float minX =  999f, minY =  999f;
        float maxX = -999f, maxY = -999f;

        for (int i = 0; i < landmarks.Length; i++)
        {
            float x = landmarks[i].x;
            float y = 1f - landmarks[i].y;   // ← flip Y
            if (float.IsNaN(x) || float.IsNaN(y)) continue;
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        if (minX > maxX || minY > maxY) return null;

        // ── Pad ───────────────────────────────────────────────────────────
        float padX = (maxX - minX) * Mathf.Max(0f, paddingPct);
        float padY = (maxY - minY) * Mathf.Max(0f, paddingPct);
        minX = Mathf.Clamp01(minX - padX);
        minY = Mathf.Clamp01(minY - padY);
        maxX = Mathf.Clamp01(maxX + padX);
        maxY = Mathf.Clamp01(maxY + padY);

        // ── Convert to pixel rect ─────────────────────────────────────────
        int x0 = Mathf.Clamp(Mathf.FloorToInt(minX * srcWidth),  0, srcWidth  - 1);
        int y0 = Mathf.Clamp(Mathf.FloorToInt(minY * srcHeight), 0, srcHeight - 1);
        int x1 = Mathf.Clamp(Mathf.CeilToInt (maxX * srcWidth),  x0 + 1, srcWidth);
        int y1 = Mathf.Clamp(Mathf.CeilToInt (maxY * srcHeight), y0 + 1, srcHeight);

        int cw = x1 - x0;
        int ch = y1 - y0;
        if (cw < 8 || ch < 8) return null;

        // ── Copy directly from the raw buffer ─────────────────────────────
        Color32[] dst = new Color32[cw * ch];
        for (int row = 0; row < ch; row++)
        {
            int srcRow = (y0 + row) * srcWidth;
            int dstRow = row * cw;
            for (int col = 0; col < cw; col++)
                dst[dstRow + col] = srcPixels[srcRow + x0 + col];
        }

        var crop = new Texture2D(cw, ch, TextureFormat.RGBA32, false);
        crop.SetPixels32(dst);
        crop.Apply(false, false);
        return crop;
    }

    /// <summary>
    /// Convenience overload — extracts pixels from the Texture2D on the CPU.
    /// Only use this if you are certain the texture is CPU-readable
    /// (created with new Texture2D and Apply(false,false)).
    /// </summary>
    public static Texture2D CropFace(
        Texture2D frame,
        Vector3[] landmarks,
        bool transparentOutside = true,   // kept for API compatibility
        float paddingPct = 0.15f
    )
    {
        if (frame == null) return null;
        return CropFace(frame.GetPixels32(), frame.width, frame.height, landmarks, paddingPct);
    }
}
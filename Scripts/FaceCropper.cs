using System;
using UnityEngine;

public static class FaceCropper
{
    /// Crops the face region using the min/max bounds of the 468 landmarks (normalized 0..1).
    public static Texture2D CropFace(
        Texture2D frame,
        Vector3[] landmarks,
        bool transparentOutside = true,
        float paddingPct = 0.15f
    )
    {
        if (frame == null) return null;
        if (landmarks == null || landmarks.Length == 0) return null;

        // Landmarks assumed normalized in [0..1] in the same orientation as 'frame'
        float minX =  999f, minY =  999f;
        float maxX = -999f, maxY = -999f;

        for (int i = 0; i < landmarks.Length; i++)
        {
            float x = landmarks[i].x;
            float y = landmarks[i].y;

            // skip obviously invalid
            if (float.IsNaN(x) || float.IsNaN(y)) continue;

            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        if (minX > maxX || minY > maxY) return null;

        // Pad in normalised space
        float wN = maxX - minX;
        float hN = maxY - minY;

        float padX = wN * Mathf.Max(0f, paddingPct);
        float padY = hN * Mathf.Max(0f, paddingPct);

        minX -= padX; maxX += padX;
        minY -= padY; maxY += padY;

        // Clamp to [0..1]
        minX = Mathf.Clamp01(minX);
        minY = Mathf.Clamp01(minY);
        maxX = Mathf.Clamp01(maxX);
        maxY = Mathf.Clamp01(maxY);

        // Convert to pixel rect
        int x0 = Mathf.FloorToInt(minX * frame.width);
        int y0 = Mathf.FloorToInt(minY * frame.height);
        int x1 = Mathf.CeilToInt (maxX * frame.width);
        int y1 = Mathf.CeilToInt (maxY * frame.height);

        x0 = Mathf.Clamp(x0, 0, frame.width - 1);
        y0 = Mathf.Clamp(y0, 0, frame.height - 1);
        x1 = Mathf.Clamp(x1, x0 + 1, frame.width);
        y1 = Mathf.Clamp(y1, y0 + 1, frame.height);

        int cw = x1 - x0;
        int ch = y1 - y0;

        if (cw < 8 || ch < 8) return null;

        Color32[] src = frame.GetPixels32();
        var crop = new Texture2D(cw, ch, TextureFormat.RGBA32, false);

        // Copy pixels
        Color32[] dst = new Color32[cw * ch];
        for (int y = 0; y < ch; y++)
        {
            int srcY = y0 + y;
            int srcRow = srcY * frame.width;
            int dstRow = y * cw;

            for (int x = 0; x < cw; x++)
            {
                int srcX = x0 + x;
                dst[dstRow + x] = src[srcRow + srcX];
            }
        }

        crop.SetPixels32(dst);
        crop.Apply(false, false);

        return crop;
    }
}

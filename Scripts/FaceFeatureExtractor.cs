using UnityEngine;

public static class FaceFeatureExtractor
{
    private const int LEFT_EYE_TOP = 159;
    private const int LEFT_EYE_BOTTOM = 145;
    private const int RIGHT_EYE_TOP = 386;
    private const int RIGHT_EYE_BOTTOM = 374;

    private const int LEFT_MOUTH_CORNER = 61;
    private const int RIGHT_MOUTH_CORNER = 291;
    private const int UPPER_LIP = 13;
    private const int LOWER_LIP = 14;

    private const int NOSE_TIP = 1;
    private const int CHIN = 152;
    private const int LEFT_CHEEK = 234;
    private const int RIGHT_CHEEK = 454;

    private const int LEFT_BROW = 105;
    private const int RIGHT_BROW = 334;
    private const int LEFT_EYE_REF = 33;
    private const int RIGHT_EYE_REF = 263;

    public static FaceFeatures From468(Vector3[] lm)
    {
        var f = new FaceFeatures();
        if (lm == null || lm.Length < 468) return f;

        float faceW = Mathf.Max(1e-6f, Dist(lm[LEFT_CHEEK], lm[RIGHT_CHEEK]));
        float faceH = Mathf.Max(1e-6f, Dist(lm[CHIN], lm[NOSE_TIP]));

        float eyeOpenL = Dist(lm[LEFT_EYE_TOP], lm[LEFT_EYE_BOTTOM]);
        float eyeOpenR = Dist(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]);
        f.eye_open = ((eyeOpenL + eyeOpenR) * 0.5f) / faceH;

        float mouthOpen = Dist(lm[UPPER_LIP], lm[LOWER_LIP]);
        float mouthWidth = Mathf.Max(1e-6f, Dist(lm[LEFT_MOUTH_CORNER], lm[RIGHT_MOUTH_CORNER]));
        f.mouth_open = mouthOpen / mouthWidth;

        float cornersAvgY = (lm[LEFT_MOUTH_CORNER].y + lm[RIGHT_MOUTH_CORNER].y) * 0.5f;
        float lipsMidY = (lm[UPPER_LIP].y + lm[LOWER_LIP].y) * 0.5f;

        float smileRaw = (cornersAvgY - lipsMidY) / faceH;

        f.smile = Mathf.Clamp01(smileRaw);

        float browEyeL = Dist(lm[LEFT_BROW], lm[LEFT_EYE_REF]) / faceH;
        float browEyeR = Dist(lm[RIGHT_BROW], lm[RIGHT_EYE_REF]) / faceH;
        float browEye = (browEyeL + browEyeR) * 0.5f;
        f.brow_raise = Mathf.Clamp01((browEye - 0.10f) / 0.08f);

        float browDist = Dist(lm[LEFT_BROW], lm[RIGHT_BROW]) / faceW;
        f.brow_furrow = Mathf.Clamp01((0.55f - browDist) / 0.15f);

        float noseToLeft = Dist(lm[NOSE_TIP], lm[LEFT_CHEEK]);
        float noseToRight = Dist(lm[NOSE_TIP], lm[RIGHT_CHEEK]);
        f.head_yaw = Mathf.Clamp((noseToLeft - noseToRight) / faceW, -1f, 1f);

        f.head_pitch = Mathf.Clamp((faceH - 0.35f) / 0.15f, -1f, 1f);

        Vector2 cL = new Vector2(lm[LEFT_CHEEK].x, lm[LEFT_CHEEK].y);
        Vector2 cR = new Vector2(lm[RIGHT_CHEEK].x, lm[RIGHT_CHEEK].y);
        float angle = Mathf.Atan2(cR.y - cL.y, cR.x - cL.x);
        f.head_roll = Mathf.Clamp(angle / 0.5f, -1f, 1f);

        f.blink_rate_10s = 0f;
        return f;
    }

    private static float Dist(Vector3 a, Vector3 b)
    {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        return Mathf.Sqrt(dx * dx + dy * dy + dz * dz);
    }
}

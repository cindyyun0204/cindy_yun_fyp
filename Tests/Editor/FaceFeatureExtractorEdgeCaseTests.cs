using NUnit.Framework;
using UnityEngine;

public class FaceFeatureExtractorEdgeCaseTests
{
    [Test]
    public void From468_DoesNotThrow_WhenAllPointsZero()
    {
        var lm = new Vector3[468]; // all zeros
        Assert.DoesNotThrow(() => FaceFeatureExtractor.From468(lm));
    }

    [Test]
    public void From468_MouthOpen_Increases_WhenLipsFartherApart()
    {
        var lm1 = new Vector3[468];
        var lm2 = new Vector3[468];

        // minimal required points
        lm1[234] = new Vector3(-1, 0, 0); lm1[454] = new Vector3(1, 0, 0);
        lm1[1] = Vector3.zero; lm1[152] = new Vector3(0, -2, 0);
        lm1[61] = new Vector3(-0.5f, -1, 0); lm1[291] = new Vector3(0.5f, -1, 0);
        lm1[13] = new Vector3(0, -0.95f, 0); lm1[14] = new Vector3(0, -1.05f, 0);

        // copy baseline into lm2
        for (int i = 0; i < 468; i++) lm2[i] = lm1[i];

        // increase lip separation
        lm2[13] = new Vector3(0, -0.7f, 0);
        lm2[14] = new Vector3(0, -1.3f, 0);

        var f1 = FaceFeatureExtractor.From468(lm1);
        var f2 = FaceFeatureExtractor.From468(lm2);

        Assert.That(f2.mouth_open, Is.GreaterThan(f1.mouth_open));
    }

    [Test]
    public void From468_Smile_IsClamped_0To1()
    {
        var lm = new Vector3[468];

        // minimal required points
        lm[234] = new Vector3(-1, 0, 0); lm[454] = new Vector3(1, 0, 0);
        lm[1] = Vector3.zero; lm[152] = new Vector3(0, -2, 0);
        lm[61] = new Vector3(-0.5f, -10, 0);  // extreme
        lm[291] = new Vector3(0.5f, -10, 0);  // extreme

        // lips just to avoid degenerate width
        lm[13] = new Vector3(0, -0.95f, 0);
        lm[14] = new Vector3(0, -1.05f, 0);

        var f = FaceFeatureExtractor.From468(lm);
        Assert.That(f.smile, Is.InRange(0f, 1f));
    }
}
using NUnit.Framework;
using UnityEngine;

public class FaceFeatureExtractorTests
{
    [Test]
    public void From468_Null_ReturnsDefault()
    {
        var f = FaceFeatureExtractor.From468(null);
        Assert.That(f.mouth_open, Is.EqualTo(0f));
        Assert.That(f.smile, Is.EqualTo(0f));
        Assert.That(f.eye_open, Is.EqualTo(0f));
    }

    [Test]
    public void From468_ValidArray_DoesNotReturnNaN()
    {
        var lm = new Vector3[468];

        // put the required points somewhere sensible
        lm[234] = new Vector3(-1, 0, 0);  // left cheek
        lm[454] = new Vector3(1, 0, 0);   // right cheek
        lm[1]   = new Vector3(0, 0, 0);   // nose tip
        lm[152] = new Vector3(0, -2, 0);  // chin
        lm[13]  = new Vector3(0, -0.9f, 0); // upper lip
        lm[14]  = new Vector3(0, -1.1f, 0); // lower lip
        lm[61]  = new Vector3(-0.5f, -1, 0); // mouth left
        lm[291] = new Vector3(0.5f, -1, 0);  // mouth right
        lm[159] = new Vector3(-0.3f, 0.2f, 0); // left eye top
        lm[145] = new Vector3(-0.3f, 0.1f, 0); // left eye bottom
        lm[386] = new Vector3(0.3f, 0.2f, 0);  // right eye top
        lm[374] = new Vector3(0.3f, 0.1f, 0);  // right eye bottom
        lm[105] = new Vector3(-0.3f, 0.4f, 0); // left brow
        lm[334] = new Vector3(0.3f, 0.4f, 0);  // right brow
        lm[33]  = new Vector3(-0.3f, 0.2f, 0); // left eye ref
        lm[263] = new Vector3(0.3f, 0.2f, 0);  // right eye ref

        var f = FaceFeatureExtractor.From468(lm);

        Assert.False(float.IsNaN(f.mouth_open));
        Assert.False(float.IsNaN(f.eye_open));
        Assert.False(float.IsNaN(f.smile));

        Assert.That(f.mouth_open, Is.GreaterThanOrEqualTo(0f));
        Assert.That(f.eye_open, Is.GreaterThanOrEqualTo(0f));
        Assert.That(f.smile, Is.InRange(0f, 1f));
    }
}
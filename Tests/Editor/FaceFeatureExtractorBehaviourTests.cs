using NUnit.Framework;
using UnityEngine;

public class FaceFeatureExtractorBehaviourTests
{
    private Vector3[] CreateBaselineFace(float faceW = 2f, float faceH = 2f)
    {
        var lm = new Vector3[468];

        // face width
        lm[234] = new Vector3(-faceW / 2f, 0f, 0f);
        lm[454] = new Vector3( faceW / 2f, 0f, 0f);

        // face height
        lm[1]   = new Vector3(0f, 0f, 0f);
        lm[152] = new Vector3(0f, -faceH, 0f);

        // mouth
        lm[61]  = new Vector3(-0.5f, -1f, 0f);
        lm[291] = new Vector3( 0.5f, -1f, 0f);
        lm[13]  = new Vector3(0f, -0.95f, 0f);
        lm[14]  = new Vector3(0f, -1.05f, 0f);

        // eyes
        lm[159] = new Vector3(-0.3f, 0.2f, 0f);
        lm[145] = new Vector3(-0.3f, 0.1f, 0f);
        lm[386] = new Vector3( 0.3f, 0.2f, 0f);
        lm[374] = new Vector3( 0.3f, 0.1f, 0f);

        // brow references
        lm[33]  = new Vector3(-0.3f, 0.15f, 0f);
        lm[263] = new Vector3( 0.3f, 0.15f, 0f);

        // brows
        lm[105] = new Vector3(-0.3f, 0.35f, 0f);
        lm[334] = new Vector3( 0.3f, 0.35f, 0f);

        return lm;
    }

    [Test]
    public void From468_ReturnsDefault_WhenInputIsNull()
    {
        var f = FaceFeatureExtractor.From468(null);

        Assert.That(f.mouth_open, Is.EqualTo(0f));
        Assert.That(f.smile, Is.EqualTo(0f));
        Assert.That(f.eye_open, Is.EqualTo(0f));
    }

    [Test]
    public void From468_MouthOpen_Increases_WhenLipsSeparate()
    {
        var closed = CreateBaselineFace();
        var open = CreateBaselineFace();

        open[13] = new Vector3(0f, -0.8f, 0f);
        open[14] = new Vector3(0f, -1.2f, 0f);

        var fClosed = FaceFeatureExtractor.From468(closed);
        var fOpen = FaceFeatureExtractor.From468(open);

        Assert.That(fOpen.mouth_open, Is.GreaterThan(fClosed.mouth_open));
    }

    [Test]
public void From468_Smile_Increases_WhenMouthCornersLift()
{
    var neutral = CreateBaselineFace();
    var smile = CreateBaselineFace();

    // Neutral mouth: corners roughly level with lips midpoint
    // Smile mouth: corners higher (less negative y)
    smile[61]  = new Vector3(-0.5f, -0.85f, 0f);
    smile[291] = new Vector3( 0.5f, -0.85f, 0f);

    // Keep lips lower so lip midpoint stays below corners
    smile[13] = new Vector3(0f, -0.95f, 0f);
    smile[14] = new Vector3(0f, -1.05f, 0f);

    var fNeutral = FaceFeatureExtractor.From468(neutral);
    var fSmile = FaceFeatureExtractor.From468(smile);

    Assert.That(fSmile.smile, Is.GreaterThan(fNeutral.smile));
}

    [Test]
    public void From468_BrowRaise_Increases_WhenBrowsMoveUp()
    {
        var low = CreateBaselineFace();
        var high = CreateBaselineFace();

        high[105] += new Vector3(0f, 0.2f, 0f);
        high[334] += new Vector3(0f, 0.2f, 0f);

        var fLow = FaceFeatureExtractor.From468(low);
        var fHigh = FaceFeatureExtractor.From468(high);

        Assert.That(fHigh.brow_raise, Is.GreaterThan(fLow.brow_raise));
    }

    [Test]
    public void From468_HeadYaw_ChangesSign_WhenNoseMovesLeftVsRight()
    {
        var left = CreateBaselineFace();
        var right = CreateBaselineFace();

        left[1]  = new Vector3(-0.3f, 0f, 0f);
        right[1] = new Vector3( 0.3f, 0f, 0f);

        var fLeft = FaceFeatureExtractor.From468(left);
        var fRight = FaceFeatureExtractor.From468(right);

        Assert.That(fLeft.head_yaw, Is.Not.EqualTo(fRight.head_yaw));
        Assert.That(fLeft.head_yaw, Is.InRange(-1f, 1f));
        Assert.That(fRight.head_yaw, Is.InRange(-1f, 1f));
    }
}
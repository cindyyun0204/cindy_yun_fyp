using System.IO;
using UnityEngine;


// Centralized log paths. Default is: Application.persistentDataPath/Logs

public static class LogPaths
{
    public const string OverrideKey = "LOGS_DIR";

    public static string LogsDir
    {
        get
        {
            // Optional override
            string overrideDir = PlayerPrefs.GetString(OverrideKey, "");
            if (!string.IsNullOrWhiteSpace(overrideDir))
            {
                try { Directory.CreateDirectory(overrideDir); }
                catch { /* ignore, fall back below */ }
                if (Directory.Exists(overrideDir)) return overrideDir;
            }

            // Portable default
            string dir = Path.Combine(Application.persistentDataPath, "Logs");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    public static string ImagesDir
    {
        get
        {
            string dir = Path.Combine(LogsDir, "images");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    public static string CombineInLogs(string fileName) => Path.Combine(LogsDir, fileName);
}

using System;
using System.IO;
using System.Text;
using UnityEngine;

public static class CsvLog
{
    private static bool _inited = false;
    private static string _path = null;

    public static void Init(string fileName, string[] header)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            fileName = "emotion_log.csv";

        string dir = LogPaths.LogsDir;
        Directory.CreateDirectory(dir);

        _path = Path.Combine(dir, fileName);

        if (!_inited || !File.Exists(_path) || new FileInfo(_path).Length == 0)
        {
            try
            {
                using (var sw = new StreamWriter(_path, false, new UTF8Encoding(false)))
                {
                    if (header != null && header.Length > 0)
                        sw.WriteLine(string.Join(",", EscapeAll(header)));
                }
                _inited = true;
                Debug.Log($"[CsvLog] Init CSV: {_path}");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[CsvLog] Init failed: {e.Message}");
            }
        }
        else
        {
            _inited = true;
        }
    }

    public static void AppendRow(params string[] cols)
    {
        if (!_inited || string.IsNullOrEmpty(_path))
        {
            Debug.LogWarning("[CsvLog] AppendRow called before Init()");
            return;
        }

        try
        {
            using (var sw = new StreamWriter(_path, true, new UTF8Encoding(false)))
            {
                sw.WriteLine(string.Join(",", EscapeAll(cols)));
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[CsvLog] AppendRow failed: {e.Message}");
        }
    }

    private static string[] EscapeAll(string[] cols)
    {
        if (cols == null) return Array.Empty<string>();
        var outCols = new string[cols.Length];
        for (int i = 0; i < cols.Length; i++)
            outCols[i] = Escape(cols[i]);
        return outCols;
    }

    private static string Escape(string s)
    {
        if (s == null) s = "";
        bool mustQuote = s.Contains(",") || s.Contains("\"") || s.Contains("\n") || s.Contains("\r");
        s = s.Replace("\"", "\"\"");
        return mustQuote ? $"\"{s}\"" : s;
    }
}

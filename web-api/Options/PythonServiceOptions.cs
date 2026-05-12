namespace web_api.Options;

public sealed class PythonServiceOptions
{
    public const string SectionName = "PythonService";

    public string BaseUrl { get; set; } = "http://127.0.0.1:8000";

    public int TimeoutSeconds { get; set; } = 30;
}

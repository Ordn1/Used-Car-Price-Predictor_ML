namespace web_api.Services;

public sealed class UpstreamServiceException : Exception
{
    public UpstreamServiceException(int statusCode, string message)
        : base(message)
    {
        StatusCode = statusCode;
    }

    public int StatusCode { get; }
}

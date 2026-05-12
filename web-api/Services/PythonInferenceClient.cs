using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using web_api.Models;

namespace web_api.Services;

public sealed class PythonInferenceClient(HttpClient httpClient) : IPythonInferenceClient
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        PropertyNameCaseInsensitive = true,
    };

    public async Task<ServiceHealth> GetHealthAsync(CancellationToken cancellationToken)
    {
        using var response = await httpClient.GetAsync("health", cancellationToken);
        return await ReadResponse<ServiceHealth>(response, cancellationToken);
    }

    public async Task<ModelInfoResponse> GetModelInfoAsync(CancellationToken cancellationToken)
    {
        using var response = await httpClient.GetAsync("model-info", cancellationToken);
        return await ReadResponse<ModelInfoResponse>(response, cancellationToken);
    }

    public async Task<PredictionResponse> PredictAsync(PredictionRequest request, CancellationToken cancellationToken)
    {
        using var response = await httpClient.PostAsJsonAsync("predict", request, JsonOptions, cancellationToken);
        return await ReadResponse<PredictionResponse>(response, cancellationToken);
    }

    private static async Task<T> ReadResponse<T>(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
        {
            var content = await response.Content.ReadFromJsonAsync<T>(JsonOptions, cancellationToken);
            if (content is null)
            {
                throw new UpstreamServiceException((int)response.StatusCode, "Python service returned an empty response.");
            }

            return content;
        }

        var errorBody = await response.Content.ReadAsStringAsync(cancellationToken);
        var detail = string.IsNullOrWhiteSpace(errorBody)
            ? "Python service request failed."
            : errorBody;
        throw new UpstreamServiceException((int)response.StatusCode, detail);
    }
}

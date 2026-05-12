using web_api.Models;

namespace web_api.Services;

public interface IPythonInferenceClient
{
    Task<ServiceHealth> GetHealthAsync(CancellationToken cancellationToken);

    Task<ModelInfoResponse> GetModelInfoAsync(CancellationToken cancellationToken);

    Task<PredictionResponse> PredictAsync(PredictionRequest request, CancellationToken cancellationToken);
}

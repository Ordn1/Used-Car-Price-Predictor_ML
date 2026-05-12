using Microsoft.Extensions.Options;
using web_api.Models;
using web_api.Options;
using web_api.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<PythonServiceOptions>(
	builder.Configuration.GetSection(PythonServiceOptions.SectionName)
);
builder.Services.AddCors(options =>
{
	options.AddDefaultPolicy(policy =>
	{
		policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod();
	});
});
builder.Services.AddHttpClient<IPythonInferenceClient, PythonInferenceClient>((serviceProvider, client) =>
{
	var options = serviceProvider.GetRequiredService<IOptions<PythonServiceOptions>>().Value;
	client.BaseAddress = new Uri(options.BaseUrl.TrimEnd('/') + "/");
	client.Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds);
});

var app = builder.Build();

app.UseCors();

app.MapGet("/", () => Results.Ok(new
{
	service = "Used Car Public API",
	version = "1.0.0",
	routes = new[] { "/api/health", "/api/model-info", "/api/predict" },
}));

app.MapGet("/api/health", async (IPythonInferenceClient client, CancellationToken cancellationToken) =>
{
	try
	{
		var health = await client.GetHealthAsync(cancellationToken);
		return Results.Ok(health);
	}
	catch (UpstreamServiceException exception)
	{
		return Results.Problem(statusCode: StatusCodes.Status502BadGateway, detail: exception.Message);
	}
});

app.MapGet("/api/model-info", async (IPythonInferenceClient client, CancellationToken cancellationToken) =>
{
	try
	{
		var modelInfo = await client.GetModelInfoAsync(cancellationToken);
		return Results.Ok(modelInfo);
	}
	catch (UpstreamServiceException exception)
	{
		return Results.Problem(statusCode: StatusCodes.Status502BadGateway, detail: exception.Message);
	}
});

app.MapPost("/api/predict", async (PredictionRequest request, IPythonInferenceClient client, CancellationToken cancellationToken) =>
{
	try
	{
		var prediction = await client.PredictAsync(request, cancellationToken);
		return Results.Ok(prediction);
	}
	catch (UpstreamServiceException exception)
	{
		var statusCode = exception.StatusCode >= 400 && exception.StatusCode < 600
			? exception.StatusCode
			: StatusCodes.Status502BadGateway;
		return Results.Problem(statusCode: statusCode, detail: exception.Message);
	}
});

app.Run();

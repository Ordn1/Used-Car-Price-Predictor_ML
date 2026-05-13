using System.Text.Json.Serialization;

namespace web_api.Models;

public sealed record PredictionRequest(
    [property: JsonPropertyName("yr_mfr")] int YrMfr,
    [property: JsonPropertyName("kms_run")] int KmsRun,
    [property: JsonPropertyName("fuel_type")] string FuelType,
    [property: JsonPropertyName("city")] string City,
    [property: JsonPropertyName("times_viewed")] int TimesViewed,
    [property: JsonPropertyName("body_type")] string BodyType,
    [property: JsonPropertyName("transmission")] string Transmission
);

public sealed record PredictionAnalytics(
    string CarAge,
    string Mileage,
    string Views,
    string MileageBand,
    string DemandBand,
    string EstimatedRange,
    string AgePenalty,
    string MileagePenalty,
    string DemandBoost,
    string Confidence,
    string MarketPulse,
    int ReferenceYear,
    int CarAgeYears,
    int KmsRunValue,
    int TimesViewedValue,
    double KmsPerYearValue,
    int AgePenaltyValue,
    double AgePenaltyRaw,
    int AgePenaltyCap,
    int MileagePenaltyValue,
    double MileagePenaltyRaw,
    int MileagePenaltyCap,
    int DemandBoostValue,
    double DemandBoostRaw,
    int DemandBoostCap,
    int ConfidenceValue,
    int ConfidenceUnclamped,
    int ConfidenceBase,
    int ConfidenceFloor,
    int ConfidenceCeiling,
    int ConfidenceAgeImpact,
    int ConfidenceMileageImpact,
    int ConfidenceDemandImpact,
    double RangeLowValue,
    double RangeHighValue
);

public sealed record PredictionResponse(
    [property: JsonPropertyName("predicted_price")] double PredictedPrice,
    [property: JsonPropertyName("analytics")] PredictionAnalytics Analytics,
    [property: JsonPropertyName("created_at")] DateTime CreatedAt
);

public sealed record ServiceHealth(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("model_loaded")] bool ModelLoaded,
    [property: JsonPropertyName("selected_model")] string? SelectedModel
);

public sealed record FeatureImportanceItem(
    [property: JsonPropertyName("feature")] string Feature,
    [property: JsonPropertyName("label")] string Label,
    [property: JsonPropertyName("importance")] double Importance,
    [property: JsonPropertyName("percentage")] double Percentage
);

public sealed record ModelInfoResponse(
    [property: JsonPropertyName("selected_model")] string SelectedModel,
    [property: JsonPropertyName("feature_columns")] List<string> FeatureColumns,
    [property: JsonPropertyName("raw_input_features")] List<string> RawInputFeatures,
    [property: JsonPropertyName("categorical_features")] List<string> CategoricalFeatures,
    [property: JsonPropertyName("available_options")] Dictionary<string, List<string>> AvailableOptions,
    [property: JsonPropertyName("metrics")] Dictionary<string, Dictionary<string, double>> Metrics,
    [property: JsonPropertyName("feature_importance")] List<FeatureImportanceItem> FeatureImportance,
    [property: JsonPropertyName("train_samples")] int? TrainSamples,
    [property: JsonPropertyName("test_samples")] int? TestSamples,
    [property: JsonPropertyName("final_r2")] double? FinalR2,
    [property: JsonPropertyName("final_mae")] double? FinalMae,
    [property: JsonPropertyName("final_mse")] double? FinalMse
);

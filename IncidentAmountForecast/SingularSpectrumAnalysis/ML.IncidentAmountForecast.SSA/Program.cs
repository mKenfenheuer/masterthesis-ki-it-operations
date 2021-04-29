using Microsoft.ML;
using System;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using ML.IncidentAmountForecast.SSA.Model;

namespace ML.IncidentAmountForecast.SSA
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext context = new MLContext();

            IDataView dataView = context.Data.LoadFromTextFile<ModelInput>(path: "data.csv", hasHeader: false, separatorChar: ';');
            IDataView trainingYearData = context.Data.FilterRowsByColumn(dataView, "Year", upperBound: 2019);
            IDataView evaluateYearData = context.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 2020);
            var forecastingPipeline = context.Forecasting.ForecastBySsa(
                    outputColumnName: "ForecastedIncidents",
                    inputColumnName: "TotalIncidents",
                    windowSize: 2,
                    seriesLength: 7,
                    trainSize: 365,
                    horizon: 7,
                    confidenceLevel: 0.5f,
                    confidenceLowerBoundColumn: "LowerBoundIncidents",
                    confidenceUpperBoundColumn: "UpperBoundIncidents");

            SsaForecastingTransformer forecaster = forecastingPipeline.Fit(trainingYearData);

            Evaluate(evaluateYearData, forecaster, context);

            var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(context);
            forecastEngine.CheckPoint(context, "ML.IncidentAmountForecast.zip");

            Forecast(evaluateYearData, 7, forecastEngine, context);
        }

        static void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
        {
            IDataView predictions = model.Transform(testData);

            IEnumerable<float> actual =
                mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
                    .Select(observed => observed.TotalIncidents);

            IEnumerable<float> forecast =
                mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
                    .Select(prediction => prediction.ForecastedIncidents[0]);

            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");
            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
        }

        static void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
        {
            ModelOutput forecast = forecaster.Predict();
            IEnumerable<string> forecastOutput =
                mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                    .Take(horizon)
                    .Select((ModelInput rental, int index) =>
                    {
                        string rentalDate = rental.IncidentDate.ToShortDateString();
                        float actualIncidents = rental.TotalIncidents;
                        float lowerEstimate = Math.Max(0, forecast.LowerBoundIncidents[index]);
                        float estimate = forecast.ForecastedIncidents[index];
                        float upperEstimate = forecast.UpperBoundIncidents[index];
                        return $"Date: {rentalDate}\n" +
                        $"Actual Incidents: {actualIncidents}\n" +
                        $"Forecast:         {estimate}\n" +
                        $"Lower Estimate:   {lowerEstimate}\n" +
                        $"Upper Estimate:   {upperEstimate}\n";
                    });
            Console.WriteLine("---------------------");
            foreach (var prediction in forecastOutput)
            {
                Console.WriteLine(prediction);
            }
        }
    }
}

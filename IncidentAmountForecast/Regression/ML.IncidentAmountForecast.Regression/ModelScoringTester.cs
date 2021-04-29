
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

using ML.IncidentAmountForecast.Regression.DataStructures;

using Microsoft.ML;

namespace ML.IncidentAmountForecast.Regression
{
    public static class ModelScoringTester
    {
        public static void VisualizeSomePredictions(MLContext mlContext,
                                                    string modelName,
                                                    string testDataLocation,
                                                    PredictionEngine<ModelInput, ModelOutput> predEngine,
                                                    int numberOfPredictions)
        {
            //Make a few prediction tests 
            // Make the provided number of predictions and compare with observed data from the test dataset
            var testData = ReadSampleDataFromCsvFile(testDataLocation, numberOfPredictions);

            for (int i = 0; i < numberOfPredictions; i++)
            {
                //Score
                var resultprediction = predEngine.Predict(testData[i]);

                Common.ConsoleHelper.PrintRegressionPredictionVersusObserved(resultprediction.ForecastedIncidents.ToString(),
                                                            testData[i].TotalIncidents.ToString());
            }

        }

        //This method is using regular .NET System.IO.File and LinQ to read just some sample data to test/predict with 
        public static List<ModelInput> ReadSampleDataFromCsvFile(string dataLocation, int numberOfRecordsToRead)
        {
            return File.ReadLines(dataLocation)
                .Skip(1)
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .Select(x => x.Split(';'))
                .Select(x => new ModelInput()
                {
                    Day = float.Parse(x[0], CultureInfo.InvariantCulture),
                    Month = float.Parse(x[1], CultureInfo.InvariantCulture),
                    Year = float.Parse(x[2], CultureInfo.InvariantCulture),
                    DayOfWeek = float.Parse(x[3], CultureInfo.InvariantCulture),
                    OneDayBefore = float.Parse(x[4], CultureInfo.InvariantCulture),
                    TwoDaysBefore = float.Parse(x[5], CultureInfo.InvariantCulture),
                    ThreeDaysBefore = float.Parse(x[6], CultureInfo.InvariantCulture),
                    FourDaysBefore = float.Parse(x[7], CultureInfo.InvariantCulture),
                    FiveDaysBefore = float.Parse(x[8], CultureInfo.InvariantCulture),
                    SixDaysBefore = float.Parse(x[9], CultureInfo.InvariantCulture),
                    TotalIncidents = float.Parse(x[10], CultureInfo.InvariantCulture),
                })
                .Take(numberOfRecordsToRead)
                .ToList();
        }
    }
}

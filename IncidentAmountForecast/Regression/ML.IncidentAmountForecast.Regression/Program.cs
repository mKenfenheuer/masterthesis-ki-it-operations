using System;
using Microsoft.ML;
using System.IO;
using ML.IncidentAmountForecast.Regression.DataStructures;
using Common;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;
using CsvHelper.Configuration;
using System.Globalization;
using CsvHelper;

namespace ML.IncidentAmountForecast.Regression
{
    class Program
    {
        private static string TrainingDataRelativePath = $"data_with_history_train.csv";
        private static string TestDataRelativePath = $"data_with_history_test.csv";

        private static string TrainingDataLocation = GetAbsolutePath(TrainingDataRelativePath);
        private static string TestDataLocation = GetAbsolutePath(TestDataRelativePath);

        static void Main(string[] args)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 0);

            // 1. Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(path: TrainingDataLocation, hasHeader: false, separatorChar: ';');
            var testDataView = mlContext.Data.LoadFromTextFile<ModelInput>(path: TestDataLocation, hasHeader: false, separatorChar: ';');

            // 2. Common data pre-process with pipeline data transformations

            // Concatenate all the numeric columns into a single features column
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                                                     nameof(ModelInput.Day), nameof(ModelInput.Month), nameof(ModelInput.Year),
                                                     nameof(ModelInput.DayOfWeek), nameof(ModelInput.OneDayBefore), nameof(ModelInput.TwoDaysBefore),
                                                     nameof(ModelInput.ThreeDaysBefore), nameof(ModelInput.FourDaysBefore), nameof(ModelInput.FiveDaysBefore),
                                                     nameof(ModelInput.SixDaysBefore))
                                         .AppendCacheCheckpoint(mlContext);
            // Use in-memory cache for small/medium datasets to lower training time. 
            // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            // (Optional) Peek data in training DataView after applying the ProcessPipeline's transformations  
            Common.ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 6);
            Common.ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainingDataView, dataProcessPipeline, 6);

            // Definition of regression trainers/algorithms to use
            //var regressionLearners = new (string name, IEstimator<ITransformer> value)[]
            (string name, IEstimator<ITransformer> value)[] regressionLearners =
            {
                ("FastTree", mlContext.Regression.Trainers.FastTree()),
                ("Poisson", mlContext.Regression.Trainers.LbfgsPoissonRegression()),
                ("SDCA", mlContext.Regression.Trainers.Sdca()),
                ("FastTreeTweedie", mlContext.Regression.Trainers.FastTreeTweedie()),
                ("FastForest", mlContext.Regression.Trainers.FastForest()),
                ("Gam", mlContext.Regression.Trainers.Gam()),
                ("LightGbm", mlContext.Regression.Trainers.LightGbm()),
                //Other possible learners that could be included
                //...GeneralizedAdditiveModelRegressor...
                //...OnlineGradientDescent... (Might need to normalize the features first)
            };

            List<TestResult> results = new List<TestResult>();

            // 3. Phase for Training, Evaluation and model file persistence
            // Per each regression trainer: Train, Evaluate, and Save a different model
            foreach (var trainer in regressionLearners)
            {
                Console.WriteLine("=============== Training the current model ===============");
                var trainingPipeline = dataProcessPipeline.Append(trainer.value);
                var trainedModel = trainingPipeline.Fit(trainingDataView);

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                IDataView predictions = trainedModel.Transform(testDataView);
                var metrics = mlContext.Regression.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                ConsoleHelper.PrintRegressionMetrics(trainer.value.ToString(), metrics);

                string modelRelativeLocation = $"../../../Models/{trainer.name}Model.zip";
                string modelPath = GetAbsolutePath(modelRelativeLocation);

                results.Add(new TestResult()
                {
                    AlgorithmName = trainer.name,
                    ModelPath = modelPath,
                    RMSE = metrics.RootMeanSquaredError,
                    MSE = metrics.MeanSquaredError,
                    MAE = metrics.MeanAbsoluteError,
                    R2 = metrics.RSquared
                });

                //Save the model file that can be used by any application
                mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);
                Console.WriteLine("The model is saved to {0}", modelPath);
            }

            // 4. Try/test Predictions with the created models
            // The following test predictions could be implemented/deployed in a different application (production apps)
            // that's why it is seggregated from the previous loop
            // For each trained model, test 10 predictions           
            foreach (var learner in regressionLearners)
            {
                //Load current model from .ZIP file
                string modelRelativeLocation = $"../../../Models/{learner.name}Model.zip";
                string modelPath = GetAbsolutePath(modelRelativeLocation);
                ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

                // Create prediction engine related to the loaded trained model
                var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

                Console.WriteLine($"================== Visualize/test 10 predictions for model {learner.name}Model.zip ==================");
                //Visualize 10 tests comparing prediction with actual/observed values from the test dataset
                ModelScoringTester.VisualizeSomePredictions(mlContext, learner.name, TestDataLocation, predEngine, 10);
            }

            results = results.OrderBy(r => r.MSE).ToList();
            var config = new CsvConfiguration(CultureInfo.CurrentUICulture)
            {
                Delimiter = ";"
            };
            using (var writer = new StreamWriter($"../../../Models/regression_experiment_results.csv"))
            using (var csv = new CsvWriter(writer, config))
            {
                csv.WriteRecords(results);
            }

            Common.ConsoleHelper.ConsolePressAnyKey();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}

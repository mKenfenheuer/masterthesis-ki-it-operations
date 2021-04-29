using System;
using System.Collections.Generic;
using Microsoft.ML;
using System.IO;
using Microsoft.ML.Data;
using System.Data.SqlClient;
using System.Linq;
using Common;
using CsvHelper;
using System.Globalization;
using CsvHelper.Configuration;

namespace ML.EventClassiciation.ClassificationExperiment
{
    class Program
    {
        static void Main(string[] args)
        {
            var labelColumns = new string[] {
                nameof(EventInput.Urgency),
                nameof(EventInput.AssignmentGroup)
            };

            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 0);

            var dataView = mlContext.Data.LoadFromTextFile<EventInput>("Data/data.csv", separatorChar: ';', hasHeader: true);

            var total = dataView.GetRowCount();

            Console.WriteLine("Loading Data into Memory");

            var incidentsData = mlContext.Data.CreateEnumerable<EventInput>(dataView, reuseRowObject: false).ToArray();

            Console.WriteLine("Data Loaded.");

            foreach (string col in labelColumns)
                TrainColumn(col, incidentsData);

            Common.ConsoleHelper.ConsolePressAnyKey();
        }

        static void TrainColumn(string labelColumn, EventInput[] incidentsData)
        {
            Console.CursorTop++;
            Console.WriteLine("Training for " + labelColumn);
            Console.CursorTop++;

            var mlContext = new MLContext(seed: 0);

            var split = mlContext.Data.LoadFromEnumerable<EventInput>(incidentsData);

            var trainingDataView = split;
            var testDataView = split;

            // 2. Common data pre-process with pipeline data transformations

            // Concatenate all the numeric columns into a single features column
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", labelColumn)
                                         .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(EventInput.Host)))
                                         .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(EventInput.Severity)))
                                         .Append(mlContext.Transforms.Text.FeaturizeText(nameof(EventInput.Problem)))
                                         .Append(mlContext.Transforms.Concatenate("Features", new string[] {
                                                                                                 nameof(EventInput.Problem),
                                                                                                 nameof(EventInput.HourOfDay),
                                                                                                 nameof(EventInput.Severity),
                                                                                                 nameof(EventInput.Host) }))
                                         .Append(mlContext.Transforms.SelectColumns(new string[] { "Label", "Features" }));
                                         //.AppendCacheCheckpoint(mlContext);


            // Use in-memory cache for small/medium datasets to lower training time. 
            // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            // Definition of regression trainers/algorithms to use
            //var regressionLearners = new (string name, IEstimator<ITransformer> value)[]
            (string name, IEstimator<ITransformer> value)[] regressionLearners =
            {
                ("SdcaMaximumEntropy", mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy()),
                ("LightGbm", mlContext.MulticlassClassification.Trainers.LightGbm()),
                ("NaiveBayes", mlContext.MulticlassClassification.Trainers.NaiveBayes()),
                ("LbfgsMaximumEntropy", mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy()),
                ("SdcaNonCalibrated", mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated()),
                ("OvsA" + "SdcaLogisticRegression", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression())),
                ("OvsA" + "SdcaNonCalibrated", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SdcaNonCalibrated())),
                ("OvsA" + "SgdCalibrated", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SgdCalibrated())),
                ("OvsA" + "SgdNonCalibrated", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SgdNonCalibrated())),
                ("OvsA" + "AveragedPerceptron", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron())),
                ("OvsA" + "FastForest", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastForest())),
                ("OvsA" + "FastTree", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree())),
                //("OvsA" + "Gam", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.Gam())),
                ("OvsA" + "LinearSvm", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LinearSvm())),
                ("OvsA" + "LightGbm", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LightGbm())),
                ("OvsA" + "LdSvm", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LdSvm())),
                //("OvsA" + "Prior", mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.Prior())),
                ("Pwc" + "SdcaLogisticRegression", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression())),
                ("Pwc" + "SgdCalibrated", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.SdcaNonCalibrated())),
                ("Pwc" + "SdcaNonCalibrated", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.SgdCalibrated())),
                ("Pwc" + "SgdNonCalibrated", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.SgdNonCalibrated())),
                ("Pwc" + "AveragedPerceptron", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.AveragedPerceptron())),
                ("Pwc" + "FastForest", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.FastForest())),
                ("Pwc" + "FastTree", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.FastTree())),
                //("Pwc" + "Gam", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.Gam())),
                ("Pwc" + "LinearSvm", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.LinearSvm())),
                ("Pwc" + "LightGbm", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.LightGbm())),
                ("Pwc" + "LdSvm", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.LdSvm())),
                //("Pwc" + "Prior", mlContext.MulticlassClassification.Trainers.PairwiseCoupling(mlContext.BinaryClassification.Trainers.Prior())),
                //Other possible learners that could be included
                //...GeneralizedAdditiveModelRegressor...
                //...OnlineGradientDescent... (Might need to normalize the features first)
            };

            // 3. Phase for Training, Evaluation and model file persistence
            // Per each regression trainer: Train, Evaluate, and Save a different model

            List<TestResult> results = new List<TestResult>();

            Console.WriteLine("------------------------------------------------------------------------------");
            Console.WriteLine("| Algorithm                      | Macro    | Micro    | Correct  | Top 5    |");
            Console.WriteLine("------------------------------------------------------------------------------");

            foreach (var trainer in regressionLearners)
            {
                string modelRelativeLocation = $"..\\..\\..\\Models\\{labelColumn}_events_{trainer.name}Model.zip";
                string modelPath = GetAbsolutePath(modelRelativeLocation);
                Console.WriteLine($"| {trainer.name,-30} |          |          |          |          |");
                Console.WriteLine("------------------------------------------------------------------------------");

                if (!File.Exists(modelPath))
                {
                    var trainingPipeline = dataProcessPipeline.Append(trainer.value)
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                    var trainedModel = trainingPipeline.Fit(trainingDataView);

                    //Save the model file that can be used by any application
                    mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);
                }

                ITransformer trainedModelTransformer = mlContext.Model.Load(modelPath, out var modelInputSchema);
                IDataView predictions = trainedModelTransformer.Transform(testDataView);
                var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                var engine = mlContext.Model.CreatePredictionEngine<EventInput, EventPrediction>(trainedModelTransformer);
                TestResult result = ConsoleHelper.GetMultiClassClassificationTestMetrics(trainer.name.ToString(),
                    mlContext.Data.CreateEnumerable<EventInput>(testDataView, reuseRowObject: false).ToArray(),
                    labelColumn,
                    engine,
                    metrics);

                ConsoleHelper.GetConfusionMatrix(metrics).Save($"..\\..\\..\\Models\\Confusion\\{labelColumn}_events_{trainer.name}_confusion_matrix.png");
                ConsoleHelper.GetNormalizedConfusionMatrix(metrics).Save($"..\\..\\..\\Models\\Confusion\\{labelColumn}_events_{trainer.name}_normalized_confusion_matrix.png");

                result.ModelPath = modelPath;
                Console.CursorTop--;
                Console.CursorTop--;
                Console.WriteLine($"| {trainer.name,-30} | {result.AccuracyMacro,-8:N4} | {result.AccuracyMicro,-8:N4} | {result.PercentCorrect,-8:N4} | {result.PercentTop5Correct,-8:N4} |");
                Console.WriteLine("------------------------------------------------------------------------------");
                results.Add(result);
            }



            // 4. Try/test Predictions with the created models
            // The following test predictions could be implemented/deployed in a different application (production apps)
            // that's why it is seggregated from the previous loop
            // For each trained model, test 10 predictions           
            /*foreach (var learner in regressionLearners)
            {
                //Load current model from .ZIP file
                string modelRelativeLocation = $"../../../Models/{learner.name}Model.zip";
                string modelPath = GetAbsolutePath(modelRelativeLocation);
                ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

                // Create prediction engine related to the loaded trained model
                var predEngine = mlContext.Model.CreatePredictionEngine<IncidentInput, IncidentPrediction>(trainedModel);

            }*/

            results = results.OrderByDescending(r => r.CorrectlyClassified).ToList();
            var config = new CsvConfiguration(CultureInfo.CurrentUICulture)
            {
                Delimiter = ";"
            };
            using (var writer = new StreamWriter($"../../../Models/{labelColumn.ToLower()}_experiment_results.csv"))
            using (var csv = new CsvWriter(writer, config))
            {
                csv.WriteRecords(results);
            }

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

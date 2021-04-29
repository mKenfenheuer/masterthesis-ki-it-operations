using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML;
using static Microsoft.ML.TrainCatalogBase;
using System.Diagnostics;
using ML.EventClassiciation.ClassificationExperiment;
using System.Drawing;

namespace Common
{
    public static class ConsoleHelper
    {
        public static void PrintPrediction(string prediction)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"Predicted : {prediction}");
            Console.WriteLine($"*************************************************");
        }

        public static void PrintRegressionPredictionVersusObserved(string predictionCount, string observedCount)
        {
            Console.WriteLine($"-------------------------------------------------");
            Console.WriteLine($"Predicted : {predictionCount}");
            Console.WriteLine($"Actual:     {observedCount}");
            Console.WriteLine($"-------------------------------------------------");
        }

        public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            Console.WriteLine($"************************************************************");
        }

        public static void PrintAnomalyDetectionMetrics(string name, AnomalyDetectionMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {name} anomaly detection model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Area Under ROC Curve:                       {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       Detection rate at false positive count: {metrics.DetectionRateAtFalsePositiveCount}");
            Console.WriteLine($"************************************************************");
        }

        public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }

        public static TestResult PrintMultiClassClassificationTestMetrics(string name, EventInput[] trainingData, string columnName, PredictionEngine<EventInput, EventPrediction> engine, MulticlassClassificationMetrics metrics)
        {
            name = name.Substring(name.LastIndexOf(".") + 1);

            Console.WriteLine($"************************************************************");
            Console.WriteLine($"    Metrics for {name}");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    Test-Metrics for {name}");
            Console.WriteLine($"*-----------------------------------------------------------");

            int totalNumbers = trainingData.Count();

            int correct = 0;
            int correctTop5 = 0;

            foreach (EventInput row in trainingData)
            {
                var prediction = engine.Predict(row);
                var output = GetScoresWithLabelsSorted(engine.OutputSchema, "Label", prediction.Score);
                var data = output.AsEnumerable().OrderByDescending(p => p.Value).ToArray();

                Func<EventInput, string> getCorrectValue = i => (string)i.GetType().GetProperty(columnName).GetValue(i, null);

                if (data.FirstOrDefault().Key == getCorrectValue(row))
                    correct++;
                if (data.Take(5).Any(p => p.Key == getCorrectValue(row)))
                    correctTop5++;
            }

            Console.WriteLine($"    Total Tests = {totalNumbers}");
            Console.WriteLine($"    Correctly classified = {correct}");
            Console.WriteLine($"    Correct in top 5 predictions = {correctTop5}");
            Console.WriteLine($"    Percent correct = {((float)correct / (float)totalNumbers * 100.0):0.####} %");
            Console.WriteLine($"    Percent correct top 5 = {((float)correctTop5 / (float)totalNumbers * 100.0):0.####} %");
            Console.WriteLine($"************************************************************");

            return new TestResult()
            {
                AlgorithmName = name,
                AccuracyMacro = metrics.MacroAccuracy,
                AccuracyMicro = metrics.MicroAccuracy,
                CorrectlyClassified = correct,
                CorrectlyTop5 = correctTop5,
                TotalTests = totalNumbers,
                PercentCorrect = ((float)correct / (float)totalNumbers * 100.0),
                PercentTop5Correct = ((float)correctTop5 / (float)totalNumbers * 100.0),
            };
        }

        public static TestResult GetMultiClassClassificationTestMetrics(string name, EventInput[] trainingData, string columnName, PredictionEngine<EventInput, EventPrediction> engine, MulticlassClassificationMetrics metrics)
        {
            name = name.Substring(name.LastIndexOf(".") + 1);

            int totalNumbers = trainingData.Count();

            int correct = 0;
            int correctTop5 = 0;

            foreach (EventInput row in trainingData)
            {
                var prediction = engine.Predict(row);
                var output = GetScoresWithLabelsSorted(engine.OutputSchema, "Label", prediction.Score);
                var data = output.AsEnumerable().OrderByDescending(p => p.Value).ToArray();

                Func<EventInput, string> getCorrectValue = i => (string)i.GetType().GetProperty(columnName).GetValue(i, null).ToString();

                if (data.FirstOrDefault().Key == getCorrectValue(row))
                    correct++;
                if (data.Take(5).Any(p => p.Key == getCorrectValue(row)))
                    correctTop5++;
            }

            return new TestResult()
            {
                AlgorithmName = name,
                AccuracyMacro = metrics?.MacroAccuracy ?? -1,
                AccuracyMicro = metrics?.MicroAccuracy ?? -1,
                CorrectlyClassified = correct,
                CorrectlyTop5 = correctTop5,
                TotalTests = totalNumbers,
                PercentCorrect = ((float)correct / (float)totalNumbers),
                PercentTop5Correct = ((float)correctTop5 / (float)totalNumbers),
            };
        }

        private static Dictionary<string, float> GetScoresWithLabelsSorted(DataViewSchema schema, string name, float[] scores)
        {
            Dictionary<string, float> result = new Dictionary<string, float>();

            var column = schema.GetColumnOrNull(name);

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column.Value.Annotations.GetValue("KeyValues", ref slotNames);
            var num = 0;
            foreach (var denseValue in slotNames.DenseValues())
            {
                result.Add(denseValue.ToString(), scores[num++]);
            }

            return result.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
        }

        public static Bitmap GetConfusionMatrix(MulticlassClassificationMetrics metrics)
        {
            double[,] matrix = new double[metrics.ConfusionMatrix.NumberOfClasses, metrics.ConfusionMatrix.NumberOfClasses];

            for (int c = 0; c < metrics.ConfusionMatrix.NumberOfClasses; c++)
            {
                for (int p = 0; p < metrics.ConfusionMatrix.NumberOfClasses; p++)
                {
                    matrix[c, p] = metrics.ConfusionMatrix.GetCountForClassPair(p, c);
                }
            }


            return DrawConfusionMatrix(matrix);
        }


        public static Bitmap GetNormalizedConfusionMatrix(MulticlassClassificationMetrics metrics)
        {
            double[,] matrix = new double[metrics.ConfusionMatrix.NumberOfClasses, metrics.ConfusionMatrix.NumberOfClasses];
            double[] matrixCount = new double[metrics.ConfusionMatrix.NumberOfClasses];

            for (int c = 0; c < metrics.ConfusionMatrix.NumberOfClasses; c++)
            {
                for (int p = 0; p < metrics.ConfusionMatrix.NumberOfClasses; p++)
                {
                    matrix[c, p] = metrics.ConfusionMatrix.GetCountForClassPair(p, c);
                    matrixCount[p] += matrix[c, p];
                }
            }


            for (int c = 0; c < metrics.ConfusionMatrix.NumberOfClasses; c++)
            {
                for (int p = 0; p < metrics.ConfusionMatrix.NumberOfClasses; p++)
                {
                    matrix[c, p] = metrics.ConfusionMatrix.GetCountForClassPair(p, c) / Math.Max(1, matrixCount[p]);
                }
            }


            return DrawConfusionMatrix(matrix);
        }

        private static Bitmap DrawConfusionMatrix(double[,] matrix)
        {
            int classes = (int)Math.Sqrt(matrix.Length);

            Size rectSizes = new Size(128, 128);
            Point legendPos = new Point((int)((classes + 1) * rectSizes.Width + rectSizes.Width * 0.2f), rectSizes.Height);

            Bitmap bitmap = new Bitmap((classes + 4) * rectSizes.Width, (classes + 1) * rectSizes.Height);

            Graphics g = Graphics.FromImage(bitmap);

            double max = matrix.Cast<double>().Max();
            double min = matrix.Cast<double>().Min();

            Color maxColor = Color.FromArgb(255, 45, 30, 62);
            Color minColor = Color.FromArgb(255, 249, 241, 236);

            Font font = new Font(FontFamily.GenericSansSerif, (int)(rectSizes.Height * 0.8), FontStyle.Regular);

            font = FindBestFitFont(g, Math.Round(max, 3, MidpointRounding.ToZero).ToString(), font, new Size(rectSizes.Width * 4 / 5, rectSizes.Width * 4 / 5));

            int totalWidth = (int)(legendPos.X + (int)(rectSizes.Width * 0.3f) + g.MeasureString(((int)max).ToString() + ".0", font).Width);

            bitmap = new Bitmap(totalWidth, (classes + 1) * rectSizes.Height);
            g = Graphics.FromImage(bitmap);

            g.Clear(Color.White);

            if (max == 1)
                font = FindBestFitFont(g, "0.99", font, rectSizes);

            for (int c = 0; c < classes; c++)
            {
                for (int p = 0; p < classes; p++)
                {
                    double value = matrix[c, p];
                    Color color = ColorInterpolator.InterpolateBetween2(minColor, maxColor, (value - min) / (max - min));
                    Color colorText = ColorInterpolator.InterpolateBetween(Color.Black, Color.White, (value - min) / (max - min));
                    Rectangle rect = new Rectangle((p + 1) * rectSizes.Width, (c + 1) * rectSizes.Height, rectSizes.Width, rectSizes.Height);
                    StringFormat sf = new StringFormat();
                    sf.LineAlignment = StringAlignment.Center;
                    sf.Alignment = StringAlignment.Center;

                    g.FillRectangle(new SolidBrush(color), rect);
                    g.DrawString(Math.Round(value, 2).ToString(), font, new SolidBrush(colorText), rect, sf);
                }
            }

            for (int c = 0; c < classes; c++)
            {
                StringFormat sf = new StringFormat();
                sf.LineAlignment = StringAlignment.Center;
                sf.Alignment = StringAlignment.Center;
                g.DrawString(c.ToString(), font, Brushes.Black, new Rectangle((c + 1) * rectSizes.Width, 0, rectSizes.Width, rectSizes.Height), sf);
                g.TranslateTransform((float)classes * rectSizes.Width / 2 + rectSizes.Width, (float)classes * rectSizes.Height / 2 + rectSizes.Height);
                g.RotateTransform(-90);
                g.TranslateTransform(-(float)classes * rectSizes.Width / 2 - rectSizes.Width, -(float)classes * rectSizes.Height / 2 - rectSizes.Height);
                g.DrawString((classes - c - 1).ToString(), font, Brushes.Black, new Rectangle((c + 1) * rectSizes.Width, 0, rectSizes.Width, rectSizes.Height), sf);
                g.TranslateTransform((float)classes * rectSizes.Width / 2 + rectSizes.Width, (float)classes * rectSizes.Height / 2 + rectSizes.Height);
                g.RotateTransform(90);
                g.TranslateTransform(-(float)classes * rectSizes.Width / 2 - rectSizes.Width, -(float)classes * rectSizes.Height / 2 - rectSizes.Height);
            }


            if (classes > 2)
            {

                for (int y = 0; y < (classes * rectSizes.Height); y++)
                {
                    double value = (classes * rectSizes.Height - y) / (double)(classes * rectSizes.Height);
                    Color color = ColorInterpolator.InterpolateBetween2(minColor, maxColor, value);

                    Rectangle rect = new Rectangle(legendPos.X, legendPos.Y + y, (int)(rectSizes.Width * 0.2f), 1);
                    g.FillRectangle(new SolidBrush(color), rect);

                    if (y % (int)(classes * rectSizes.Height / 3) == 0)
                    {
                        double val = (double)(max * value);

                        StringFormat sf = new StringFormat();
                        sf.LineAlignment = StringAlignment.Center;
                        sf.Alignment = StringAlignment.Near;
                        if (y == 0)
                        {
                            val = (double)(max * (classes * (double)rectSizes.Height - y - rectSizes.Height / 2) / (double)(classes * rectSizes.Height));
                            g.DrawString(Math.Round(val, 1).ToString(), font, Brushes.Black, new Rectangle(legendPos.X + (int)(rectSizes.Width * 0.3f), legendPos.Y + y, bitmap.Width - legendPos.X + (int)(rectSizes.Width * 0.22f), rectSizes.Height), sf);
                        }
                        else if (y >= (classes * rectSizes.Height) / 3.0 * 2.0)
                        {
                            val = (double)(max * (classes * (double)rectSizes.Height - y + rectSizes.Height / 2) / (double)(classes * rectSizes.Height));
                            g.DrawString(Math.Round(val, 1).ToString(), font, Brushes.Black, new Rectangle(legendPos.X + (int)(rectSizes.Width * 0.3f), legendPos.Y + y - rectSizes.Height, bitmap.Width - legendPos.X + (int)(rectSizes.Width * 0.22f), rectSizes.Height), sf);
                        }
                        else
                        {
                            g.DrawString(Math.Round(val, 1).ToString(), font, Brushes.Black, new Rectangle(legendPos.X + (int)(rectSizes.Width * 0.3f), legendPos.Y + y - rectSizes.Height / 2, bitmap.Width - legendPos.X + (int)(rectSizes.Width * 0.22f), rectSizes.Height), sf);
                        }
                    }
                }
            }
            else
            {
                for (int y = 0; y < (classes * rectSizes.Height); y++)
                {
                    double value = (classes * rectSizes.Height - y) / (double)(classes * rectSizes.Height);
                    Color color = ColorInterpolator.InterpolateBetween2(minColor, maxColor, value);

                    Rectangle rect = new Rectangle(legendPos.X, legendPos.Y + y, (int)(rectSizes.Width * 0.2f), 1);
                    g.FillRectangle(new SolidBrush(color), rect);
                }

                StringFormat sf = new StringFormat();
                sf.LineAlignment = StringAlignment.Center;
                sf.Alignment = StringAlignment.Near;
                double val = (double)(max);
                g.DrawString(Math.Round(val, 1).ToString(), font, Brushes.Black, new Rectangle(legendPos.X + (int)(rectSizes.Width * 0.3f), legendPos.Y, bitmap.Width - legendPos.X + (int)(rectSizes.Width * 0.22f), rectSizes.Height), sf);
                val = (double)(0);
                g.DrawString(Math.Round(val, 1).ToString(), font, Brushes.Black, new Rectangle(legendPos.X + (int)(rectSizes.Width * 0.3f), legendPos.Y + (classes - 1) * rectSizes.Height, bitmap.Width - legendPos.X + (int)(rectSizes.Width * 0.22f), rectSizes.Height), sf);

            }

            g.Save();

            return bitmap;
        }

        private static Font FindBestFitFont(Graphics g, String text, Font font, Size proposedSize)
        {
            // Compute actual size, shrink if needed
            while (true)
            {
                SizeF size = g.MeasureString(text, font);

                // It fits, back out
                if (size.Height <= proposedSize.Height &&
                     size.Width <= proposedSize.Width) { return font; }

                // Try a smaller font (90% of old size)
                Font oldFont = font;
                font = new Font(font.Name, (float)(font.Size * .9), font.Style);
                oldFont.Dispose();
            }
        }

        public static void PrintRegressionFoldsAverageMetrics(string algorithmName, IReadOnlyList<CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:    {L1.Average():0.###} ");
            Console.WriteLine($"*       Average L2 Loss:    {L2.Average():0.###}  ");
            Console.WriteLine($"*       Average RMS:          {RMS.Average():0.###}  ");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared: {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(
                                         string algorithmName,
                                       IReadOnlyList<CrossValidationResult<MulticlassClassificationMetrics>> crossValResults
                                                                           )
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for {algorithmName} Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");

        }

        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }

        public static void PrintClusteringMetrics(string name, ClusteringMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} clustering model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Average Distance: {metrics.AverageDistance}");
            Console.WriteLine($"*       Davies Bouldin Index is: {metrics.DaviesBouldinIndex}");
            Console.WriteLine($"*************************************************");
        }

        public static void ShowDataViewInConsole(MLContext mlContext, IDataView dataView, int numberOfRows = 4)
        {
            string msg = string.Format("Show data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            var preViewTransformedData = dataView.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        [Conditional("DEBUG")]
        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        public static void PeekDataViewInConsole(MLContext mlContext, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            //https://github.com/dotnet/machinelearning/blob/master/docs/code/MlNetCookBook.md#how-do-i-look-at-the-intermediate-data
            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // 'transformedData' is a 'promise' of data, lazy-loading. call Preview  
            //and iterate through the returned collection from preview.

            var preViewTransformedData = transformedData.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                Console.WriteLine(lineToPrint + "\n");
            }
        }

        [Conditional("DEBUG")]
        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        public static void PeekVectorColumnDataInConsole(MLContext mlContext, string columnName, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: : Show {0} rows with just the '{1}' column", numberOfRows, columnName);
            ConsoleWriteHeader(msg);

            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // Extract the 'Features' column.
            var someColumnData = transformedData.GetColumn<float[]>(columnName)
                                                        .Take(numberOfRows).ToList();

            // print to console the peeked rows

            int currentRow = 0;
            someColumnData.ForEach(row =>
            {
                currentRow++;
                String concatColumn = String.Empty;
                foreach (float f in row)
                {
                    concatColumn += f.ToString();
                }

                Console.WriteLine();
                string rowMsg = string.Format("**** Row {0} with '{1}' field value ****", currentRow, columnName);
                Console.WriteLine(rowMsg);
                Console.WriteLine(concatColumn);
                Console.WriteLine();
            });
        }

        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new string('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsoleWriterSection(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new string('-', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" ");
            Console.WriteLine("Press any key to finish.");
            Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            const string exceptionTitle = "EXCEPTION";
            Console.WriteLine(" ");
            Console.WriteLine(exceptionTitle);
            Console.WriteLine(new string('#', exceptionTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }

        public static void ConsoleWriteWarning(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.DarkMagenta;
            const string warningTitle = "WARNING";
            Console.WriteLine(" ");
            Console.WriteLine(warningTitle);
            Console.WriteLine(new string('#', warningTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }
    }
}
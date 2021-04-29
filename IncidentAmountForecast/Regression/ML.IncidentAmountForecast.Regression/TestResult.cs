using System;
using System.Collections.Generic;
using System.Text;

namespace ML.IncidentAmountForecast.Regression
{
    public class TestResult
    {
        public string AlgorithmName { get; set; }
        public string ModelPath { get; set; }
        public double MSE { get; set; }
        public double RMSE { get; set; }
        public double MAE { get; set; }
        public double R2 { get; set; }
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace ML.EventClassiciation.ClassificationExperiment
{
    public class TestResult
    {
        public string AlgorithmName { get; set; }
        public string ModelPath { get; set; }
        public double AccuracyMacro { get; set; }
        public double AccuracyMicro { get; set; }
        public int TotalTests { get; set; }
        public int CorrectlyClassified { get; set; }
        public int CorrectlyTop5 { get; set; }
        public double PercentCorrect { get; set; }
        public double PercentTop5Correct { get; set; }
    }
}

// This file was auto-generated by ML.NET Model Builder. 
using System;

using Microsoft.ML.Data;

namespace ML.IncidentAmountForecast.Regression.DataStructures
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public float Day { get; set; }

        [LoadColumn(1)]
        public float Month { get; set; }

        [LoadColumn(2)]
        public float Year { get; set; }

        [LoadColumn(3)]
        public float DayOfWeek { get; set; }

        [LoadColumn(4)]
        public float SixDaysBefore { get; set; }

        [LoadColumn(5)]
        public float FiveDaysBefore { get; set; }

        [LoadColumn(6)]
        public float FourDaysBefore { get; set; }

        [LoadColumn(7)]
        public float ThreeDaysBefore { get; set; }

        [LoadColumn(8)]
        public float TwoDaysBefore { get; set; }

        [LoadColumn(9)]
        public float OneDayBefore { get; set; }

        [LoadColumn(10)]
        [ColumnName("Label")]
        public float TotalIncidents { get; set; }
    }
}
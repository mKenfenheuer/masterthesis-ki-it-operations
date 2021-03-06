// This file was auto-generated by ML.NET Model Builder. 
using System;

using Microsoft.ML.Data;

namespace ML.IncidentAmountForecast.SSA.Model
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public DateTime IncidentDate { get; set; }

        [LoadColumn(1)]
        public float Year { get; set; }

        [LoadColumn(2)]
        public float DayOfWeek { get; set; }

        [LoadColumn(3)]
        public float TotalIncidents { get; set; }
    }
}

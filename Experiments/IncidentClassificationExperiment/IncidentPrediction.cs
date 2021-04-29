using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace ML.IncidentClassification.ClassificationExperiment
{
    public class IncidentPrediction
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public int Prediction { get; set; }
        public float[] Score { get; set; }
    }
}

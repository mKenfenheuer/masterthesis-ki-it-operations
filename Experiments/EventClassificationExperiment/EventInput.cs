using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace ML.EventClassiciation.ClassificationExperiment
{
    public class EventInput
    {
        [ColumnName("Severity"), LoadColumn(0)]
        public string Severity { get; set; }


        [ColumnName("Time"), LoadColumn(1)]
        public string Time { get; set; }


        [ColumnName("HourOfDay"), LoadColumn(2)]
        public float HourOfDay { get; set; }

        [ColumnName("Host"), LoadColumn(3)]
        public string Host { get; set; }


        [ColumnName("Problem"), LoadColumn(4)]
        public string Problem { get; set; }


        [ColumnName("Urgency"), LoadColumn(5)]
        public string Urgency { get; set; }


        [ColumnName("AssignmentGroup"), LoadColumn(6)]
        public string AssignmentGroup { get; set; }
    }
}

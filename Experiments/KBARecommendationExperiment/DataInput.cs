using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace ML.KBARecommendation.ClassificationExperiment
{
    public class DataInput
    {
        [LoadColumn(0)]
        public int Number { get; set; }


        [LoadColumn(1)]
        public string Title { get; set; }


        [LoadColumn(2)]
        public string KBANumber { get; set; }


        [LoadColumn(3)]
        public string Resolution { get; set; }


        [LoadColumn(4)]
        public int Class { get; set; }
    }
}

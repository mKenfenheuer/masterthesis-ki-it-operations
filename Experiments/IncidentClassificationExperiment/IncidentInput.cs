using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace ML.IncidentClassification.ClassificationExperiment
{
    public class IncidentInput
    {
        [LoadColumn(0)]
        public string Title { get; set; }


        [LoadColumn(1)]
        public string Body { get; set; }


        [LoadColumn(2)]
        public int TicketType { get; set; }


        [LoadColumn(3)]
        public int Category { get; set; }


        [LoadColumn(4)]
        public int SubCategory1 { get; set; }


        [LoadColumn(5)]
        public int SubCategory2 { get; set; }


        [LoadColumn(6)]
        public int BusinessService { get; set; }


        [LoadColumn(7)]
        public int Urgency { get; set; }

        [LoadColumn(8)]
        public int Impact { get; set; }
    }
}

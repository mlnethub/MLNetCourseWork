using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SpamDetection
{
    /// <summary>
    /// The SpamPrediction class contains one single spam prediction.
    /// </summary>
    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")] public bool IsSpam { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}

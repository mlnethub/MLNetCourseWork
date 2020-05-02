using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BikeDemandPrediction
{
    /// <summary>
    /// The DemandPrediction class holds one single bike demand prediction.
    /// </summary>
    public class DemandPrediction
    {
        [ColumnName("Score")]
        public float PredictedCount;
    }
}

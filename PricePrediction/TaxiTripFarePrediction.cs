using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TaxiFarePrediction
{
    /// <summary>
    /// The TaxiTripFarePrediction class represents a single far prediction.
    /// </summary>
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
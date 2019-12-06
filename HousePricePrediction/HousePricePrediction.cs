using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousePricePrediction
{
    /// <summary>
    /// The TaxiTripFarePrediction class represents a single far prediction.
    /// </summary>
    public class HousePricePrediction
    {
        [ColumnName("Score")]
        public float SalePrice;
    }
}
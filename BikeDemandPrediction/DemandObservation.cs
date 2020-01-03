using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BikeDemandPrediction
{
    /// <summary>
    /// The DemandObservation class holds one single bike demand observation record.
    /// </summary>
    public class DemandObservation
    {
        [LoadColumn(2)] public float Season { get; set; }
        [LoadColumn(3)] public float Year { get; set; }
        [LoadColumn(4)] public float Month { get; set; }
        [LoadColumn(5)] public float Hour { get; set; }
        [LoadColumn(6)] public float Holiday { get; set; }
        [LoadColumn(7)] public float Weekday { get; set; }
        [LoadColumn(8)] public float WorkingDay { get; set; }
        [LoadColumn(9)] public float Weather { get; set; }
        [LoadColumn(10)] public float Temperature { get; set; }
        [LoadColumn(11)] public float NormalizedTemperature { get; set; }
        [LoadColumn(12)] public float Humidity { get; set; }
        [LoadColumn(13)] public float Windspeed { get; set; }
        [LoadColumn(16)] [ColumnName("Label")] public float Count { get; set; }
    }
}

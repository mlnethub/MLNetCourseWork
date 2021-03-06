using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TaxiFarePrediction
{


    /// <summary>
    /// The TaxiTrip class represents a single taxi trip.
    /// </summary>
    public class TaxiTrip
    {
        [LoadColumn(0)] public string VendorId;
        [LoadColumn(5)] public string RateCode;
        [LoadColumn(3)] public float PassengerCount;
        [LoadColumn(4)] public float TripDistance;
        [LoadColumn(9)] public string PaymentType;
        [LoadColumn(10)] public float FareAmount;
    }
}
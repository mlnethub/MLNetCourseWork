using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousePricePrediction
{
    public class HousePriceData
    {
        [LoadColumn(4)] public float LotArea;
       // [LoadColumn(13)] public string Condition1;
       // [LoadColumn(15)] public string BldgType;
       // [LoadColumn(16)] public string HouseStyle;
        [LoadColumn(19)] public float YearBuilt;
        [LoadColumn(51)] public float BedroomAbvGr;
      //  [LoadColumn(78)] public string SaleType;
       // [LoadColumn(79)] public string SaleCondition;
        [LoadColumn(80)] public float SalePrice;

    }
}
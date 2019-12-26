using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TitanicPrediction
{
    public class PaxTrip
    { //"PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"
        //[LoadColumn(0)] public float PassengerId { get; set; }
        public bool Label { get; set; }
        public float Pclass { get; set; }
        public string Name { get; set; }
        public string Sex { get; set; }
        public string RawAge { get; set; }
        public float SibSp { get; set; }
        public float Parch { get; set; }
        public string Ticket { get; set; }
        public float Fare { get; set; }
        public string Cabin { get; set; }
        public string Embarked { get; set; }

    }
    public class PaxTripFarePrediction
    {
        [ColumnName("PredictedLabel")] public bool Prediction;
        public float Probability;
        public float Score;
    }
    /// <summary>
    /// The RawAge class is a helper class for a column transformation.
    /// </summary>
    public class FromAge
    {
        public string RawAge;
    }

    /// <summary>
    /// The ProcessedAge class is a helper class for a column transformation.
    /// </summary>
    public class ToAge
    {
        public string Age;
    }
}
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DigitRecognition
{
    /// <summary>
    /// The DigitPrediction class represents one digit prediction.
    /// </summary>
    class DigitPrediction
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}

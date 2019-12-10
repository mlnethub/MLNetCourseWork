using Microsoft.ML;
using Microsoft.ML.Data;

namespace DigitRecognition
{
    /// <summary>
    /// The Digit class represents one mnist digit.
    /// </summary>
    class Digit
    {
        [ColumnName("PixelValues")]
        [VectorType(784)]
        public float[] PixelValues;

        [LoadColumn(0)]
        public float Number;
    }
}

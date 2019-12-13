using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SpamDetection
{
    /// <summary>
    /// The SpamInput class contains one single message which may be spam or ham.
    /// </summary>
    public class SpamInput
    {
        [LoadColumn(0)] public string RawLabel { get; set; }
        [LoadColumn(1)] public string Message { get; set; }
    }

    /// <summary>
    /// This class describes which input columns we want to transform.
    /// </summary>
    public class FromLabel
    {
        public string RawLabel { get; set; }
    }

    /// <summary>
    /// This class describes what output columns we want to produce.
    /// </summary>
    public class ToLabel
    {
        public bool Label { get; set; }
    }
}

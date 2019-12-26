using PLplot;
using System;
using System.Reflection;
using System.IO;
using System.Linq;
using Microsoft.ML;
using BetterConsoleTables;
using Microsoft.ML.Data;

namespace DetectToxicComments
{

    class Comment
    {
        [LoadColumn(1)]
        public string Text { get; set; }

        [LoadColumn(2, 7)]
        //[VectorType(7)]
        [ColumnName("Label")]
        public float[] ClassLabel { get; set; }
    }
    class Program
    {
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "train.csv");
        static void Main(string[] args)
        {
            //Create MLContext
            MLContext mlContext = new MLContext();
            //Load Data
            var data = mlContext.Data.LoadFromTextFile<Comment>("train.csv", separatorChar: ',', hasHeader: true);
            var textEstimator = mlContext.Transforms.Text.FeaturizeText("Text");
            // Fit data to estimator
            // Fitting generates a transformer that applies the operations of defined by estimator
            ITransformer textTransformer = textEstimator.Fit(data);
            var transformedData = textTransformer.Transform(data);
            var preview = transformedData.Preview(maxRows: 10);
            // show the preview
            WritePreview(preview);
            //WritePreview(data);
        }

        /// <summary>
        /// Helper method to write a single column preview to the console.
        /// </summary>
        /// <param name="preview">The data preview to write.</param>
        /// <param name="column">The name of the column to write.</param>
        public static void WritePreviewColumn(DataDebuggerPreview preview, string column)
        {
            // set up a console table
            var table = new Table(TableConfiguration.Unicode(), new ColumnHeader(column));

            // fill the table with results
            foreach (var row in preview.RowView)
            {
                foreach (var col in row.Values)
                {
                    if (col.Key == column)
                    {
                        var vector = (VBuffer<float>)col.Value;
                        table.AddRow(string.Concat(vector.DenseValues()));
                    }
                }
            }

            // write the table
            Console.WriteLine(table.ToString());
        }

        /// <summary>
        /// Helper method to write the machine learning pipeline to the console.
        /// </summary>
        /// <param name="preview">The data preview to write.</param>
        public static void WritePreview(DataDebuggerPreview preview)
        {
            // set up a console table
            var table = new Table(
                TableConfiguration.Unicode(),
                (from c in preview.ColumnView
                 select new ColumnHeader(c.Column.Name)).ToArray());

            // fill the table with results
            foreach (var row in preview.RowView)
            {
                table.AddRow((from c in row.Values
                              select c.Value is VBuffer<float> ? "<vector>" : c.Value
                            ).ToArray());
            }

            // write the table
            Console.WriteLine(table.ToString());
        }
    }
}

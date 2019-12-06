using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.FastTree;
using Microsoft.ML.Transforms;
using PLplot;
using System.Reflection;
using System.Linq;
using Microsoft.ML.Transforms.Text;
using BetterConsoleTables;

namespace FraudDetection
{
    class Program
    {
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "creditcard.csv");

        public static bool SafeParse(string value)
        {
            var s = (value).Trim().ToLower();
            return s.Equals("0");
        }

        static void Main(string[] args)
        {
             // set up a machine learning context
            var context = new MLContext();

            // load data
            Console.WriteLine("Loading data...");
            var data = context.Data.LoadFromTextFile<CreditCardData>(dataPath, hasHeader: true, separatorChar: ',');

            // keep only transactions below 2000
            data = context.Data.FilterRowsByColumn(
                data,
                "Amount",
                upperBound: 2500
            );

            // Plot the data
            Console.WriteLine("Plotting...");
            PlotData(context, data);

            // split the data into a training and test partition
            var partitions = context.Data.TrainTestSplit(data, testFraction: 0.2);
            // set up a training pipeline
            // step 1: convert the label value to a boolean
            // var pipeline = context.Transforms.CustomMapping<FromLabel, ToLabel>(
            //         (input, output) => { output.Label = Convert.ToBoolean(Convert.ToInt32(input.ClassLabel)); },
            //         "LabelMapping"
            //     )

            var pipeline = context.Transforms.CustomMapping<CreditCardData, ToNormalizedAmount>(
                    (input, output) => { output.NormalizedAmount = input.Amount / 100; },
                    contractName: "NormalizedAmount"
                )

                .Append(context.Transforms.Conversion.ConvertType("ClassLabel", outputKind: DataKind.Boolean))
                .Append(context.Transforms.Conversion.MapValueToKey("ClassLabel"))

                .Append(context.Transforms.NormalizeBinning(
                    inputColumnName: "Time",
                    outputColumnName: "BinnedTime",
                    maximumBinCount: 1800
                ))

                .Append(context.Transforms.Categorical.OneHotEncoding(
                    inputColumnName: "BinnedTime",
                    outputColumnName: "EncodedTime"
                ))

                // step 2: concatenate all feature columns
                .Append(context.Transforms.Concatenate(
                "Features",
                "EncodedTime",
                "V1","V2","V3","V4","V5",
                "V6","V7","V8","V9","V10","V11","V12","V13",
                "V14","V15","V16","V17","V18","V19","V20",
                "V21","V22","V23","V24","V25","V26","V27",
                "V28",
                "NormalizedAmount"))

                 .Append(context.Transforms.DropColumns(
                 "Time",
                 "BinnedTime",
                 "Amount"
                ))


                .Append(context.Transforms.NormalizeSupervisedBinning("Features", fixZero: false, maximumBinCount: 5, labelColumnName: "ClassLabel"))
                .Append(context.Transforms.Categorical.OneHotEncoding("Features", outputKind: OneHotEncodingEstimator.OutputKind.Indicator))
                .Append(context.Transforms.Conversion.MapKeyToValue("ClassLabel"))

                // step 3: set up a fast tree learner
                // .Append(context.BinaryClassification.Trainers.FastTree( labelColumnName: "ClassLabel", featureColumnName: "Features"));

               .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "ClassLabel", featureColumnName: "Features"));

            // train the model
            Console.WriteLine("Training model...");
            var model = pipeline.Fit(partitions.TrainSet);

            // make predictions for the test data set
            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(partitions.TestSet);



            // Preview the data
        //    Console.WriteLine("Loading preview...");
        //    var preview = predictions.Preview(maxRows: 5);
        //    WritePreviewColumn(preview, "NormalizedAmount");

            // compare the predictions with the ground truth
            var metrics = context.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "ClassLabel",
                scoreColumnName: "Score");

            // report the results
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall}");
            Console.WriteLine();
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

        static void PlotData(MLContext context, IDataView data)
        {
            var transactions = context.Data.CreateEnumerable<CreditCardData>(data, reuseRowObject: false).ToArray();
            // plot median house value by longitude
            var pl = new PLStream();
            pl.sdev("pngcairo");                // png rendering
            pl.sfnam("data.png");               // output filename
            pl.spal0("cmap0_alternate.pal");    // alternate color palette
            pl.init();
            pl.env(
                0, 5000,                           // x-axis range
                0, 2500,                          // y-axis range
                AxesScale.Independent,          // scale x and y independently
                AxisBox.BoxTicksLabelsAxes);    // draw box, ticks, and num ticks
            pl.lab(
                "Time",                // x-axis label
                "Amount",           // y-axis label
                "Transactions by Time");    // plot title
            pl.sym(
                transactions.Select(txn => (double)txn.Time).ToArray(),
                transactions.Select(txn => (double)txn.Amount).ToArray(),
                (char)218
            );
            pl.eop();
        }
    }
}

using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using BetterConsoleTables;

namespace DigitRecognition
{
    class Program
    {
        // filenames for data set
        private static string trainDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_train.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_test.csv");

        /// <summary>
        /// The program entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create a machine learning context
            var context = new MLContext();

            // load data
            Console.WriteLine("Loading data....");
            var columnDef = new TextLoader.Column[]
            {
            new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
            new TextLoader.Column("Number", DataKind.Single, 0)
            };
            var trainDataView = context.Data.LoadFromTextFile(
                path: trainDataPath,
                columns: columnDef,
                hasHeader: true,
                separatorChar: ',');
            var testDataView = context.Data.LoadFromTextFile(
                path: testDataPath,
                columns: columnDef,
                hasHeader: true,
                separatorChar: ',');

            /*
                MapValueToKey - reads the Number column and builds a dictionary of unique values. It then produces an output column called Label which contains the dictionary key 
                for each number value. We need this step because we can only train a multiclass classifier on keys.
                Concatenate - converts the PixelValue vector into a single column called Features. This is a required step because ML.NET can only train on a single input column.
                AppendCacheCheckpoint - caches all training data at this point. This is an optimization step that speeds up the learning algorithm which comes next.
                A SdcaMaximumEntropy classification learner - will train the model to make accurate predictions.
                MapKeyToValue step - converts the keys in the Label column back to the original number values. We need this step to show the numbers when making predictions.
            */

            // build a training pipeline
            // step 1: map the number column to a key value and store in the label column
            var pipeline = context.Transforms.Conversion.MapValueToKey(
                outputColumnName: "Label",
                inputColumnName: "Number",
                keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)

                // step 2: concatenate all feature columns
                .Append(context.Transforms.Concatenate(
                    "Features",
                    nameof(Digit.PixelValues)))

                // step 3: cache data to speed up training                
                .AppendCacheCheckpoint(context)

                // step 4: train the model with SDCA
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "Label",
                    featureColumnName: "Features"))

                // step 5: map the label key value back to a number
                .Append(context.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "Number",
                    inputColumnName: "Label"));

            // train the model
            Console.WriteLine("Training model....");
            var model = pipeline.Fit(trainDataView);

            // use the model to make predictions on the test data
            Console.WriteLine("Evaluating model....");
            var predictions = model.Transform(testDataView);

            // evaluate the predictions
            var metrics = context.MulticlassClassification.Evaluate(
                data: predictions,
                labelColumnName: "Number",
                scoreColumnName: "Score");

            // show evaluation metrics
            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"    MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"    LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"    LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();
            // grab three digits from the test data
            var digits = context.Data.CreateEnumerable<Digit>(testDataView, reuseRowObject: false).ToArray();
                                     //.Take(4).ToArray();
            
            var testDigits = new Digit[] { digits[5], digits[16], digits[28], digits[63], digits[129] };

            // create a prediction engine
            var engine = context.Model.CreatePredictionEngine<Digit, DigitPrediction>(model);

            // set up a table to show the predictions
            var table = new Table(TableConfiguration.Unicode());
            table.AddColumn("Digit");
            for (var i = 0; i < 10; i++)
                table.AddColumn($"P{i}");

            // predict each test digit
            for (var i = 0; i < testDigits.Length; i++)
            {
                var prediction = engine.Predict(testDigits[i]);
                table.AddRow(
                    testDigits[i].Number,
                    prediction.Score[0].ToString("P2"),
                    prediction.Score[1].ToString("P2"),
                    prediction.Score[2].ToString("P2"),
                    prediction.Score[3].ToString("P2"),
                    prediction.Score[4].ToString("P2"),
                    prediction.Score[5].ToString("P2"),
                    prediction.Score[6].ToString("P2"),
                    prediction.Score[7].ToString("P2"),
                    prediction.Score[8].ToString("P2"),
                    prediction.Score[9].ToString("P2"));
            }

            // show results
            Console.WriteLine(table.ToString());

        }
    }
}

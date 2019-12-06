using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousePricePrediction
{
    class Program
    {
        static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "data.csv");
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // set up the text loader
            var textLoader = mlContext.Data.CreateTextLoader(
                new TextLoader.Options()
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    Columns = new[]
                    {
                        new TextLoader.Column("LotArea", DataKind.Single, 4),
                       // new TextLoader.Column("BldgType", DataKind.String, 15),
                        new TextLoader.Column("BedroomAbvGr", DataKind.Single, 51),
                        new TextLoader.Column("YearBuilt", DataKind.Single, 19),
                       // new TextLoader.Column("SaleType", DataKind.String, 78),
                       // new TextLoader.Column("SaleCondition", DataKind.String, 79),
                        new TextLoader.Column("SalePrice", DataKind.Single, 80)
                    }
                }
            );

            // load the data
            Console.Write("Loading training data....");
            var dataView = textLoader.Load(dataPath);
            Console.WriteLine("done");

            // split into a training and test partition
            var partitions = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // set up a learning pipeline
            var pipeline = mlContext.Transforms.CustomMapping<HousePriceData, ToNormalizedLotArea>(
                (input, output) => { output.NormalizedLotArea = input.LotArea / 1000; },
                contractName: "NormalizedLotArea")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    inputColumnName: "YearBuilt",
                    outputColumnName: "EncodedYearBuilt"))
                .Append(mlContext.Transforms.CustomMapping<HousePriceData, ToNormalizedHousePrice>(
                (input, output) => { output.NormalizedHousePrice = input.SalePrice / 10000; },
                contractName: "NormalizedHousePrice"))
                // combine all input features into a single column
                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    "NormalizedLotArea",
                    "EncodedYearBuilt",
                    "BedroomAbvGr",
                    "NormalizedHousePrice"))
                 .Append(mlContext.Transforms.DropColumns(
                    "SalePrice",
                    "LotArea",
                    "YearBuilt"
                ))
                .Append(mlContext.Transforms.CopyColumns(
                    inputColumnName: "NormalizedHousePrice",
                    outputColumnName: "Label"
                ))
                // cache the data to speed up training
                .AppendCacheCheckpoint(mlContext)

                // use the fast tree learner
                .Append(mlContext.Regression.Trainers.FastTree());


            // train the model
            Console.Write("Training the model....");
            var model = pipeline.Fit(partitions.TrainSet);
            Console.WriteLine("done");

            // get a set of predictions
            Console.Write("Evaluating the model....");
            var predictions = model.Transform(partitions.TestSet);

            // get regression metrics to score the model
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine("done");

            // show the metrics
            Console.WriteLine();
            Console.WriteLine($"Model metrics:");
            Console.WriteLine($"  RMSE:{metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"  MSE: {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"  MAE: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine();

            // create a prediction engine for one single prediction
            var predictionFunction = mlContext.Model.CreatePredictionEngine<HousePriceData, HousePricePrediction>(model);

            // prep a single taxi trip
            var houseSample = new HousePriceData()
            {
                LotArea = 10791,
                YearBuilt = 2009,
                BedroomAbvGr = 3,
                SalePrice = 0 // the model will predict the actual fare for this trip
            };

            // make the prediction
            var prediction = predictionFunction.Predict(houseSample);

            // sho the prediction
            Console.WriteLine($"Single prediction:");
            Console.WriteLine($"  Predicted price: {prediction.SalePrice:0.####}");
        }
    }
}

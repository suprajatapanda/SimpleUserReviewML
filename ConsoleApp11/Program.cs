using Microsoft.ML;
using Microsoft.ML.Data;

public class Program
{
    public static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Load Data
        var dataPath = "C:\\Users\\supra\\source\\repos\\ConsoleApp11\\ConsoleApp11\\data.csv";
        IDataView dataView = mlContext.Data.LoadFromTextFile<TextData>(dataPath, hasHeader: true, separatorChar: '|');

        // Data process configuration with pipeline data transformations
        var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: nameof(TextData.Label))
        .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(TextData.Text)));

        var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "Features");
        var trainingPipeline = dataProcessPipeline.Append(trainer)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Train the model
        var trainedModel = trainingPipeline.Fit(dataView);

        // Evaluate the model
        var predictions = trainedModel.Transform(dataView);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "LabelKey");
        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}; MacroAccuracy: {metrics.MacroAccuracy}");

        // Prediction engine
        var predictor = mlContext.Model.CreatePredictionEngine<TextData, TextPrediction>(trainedModel);
        var review = new TextData { Text = Console.ReadLine() ?? "No Review" };
        var prediction = predictor.Predict(review);
        Console.WriteLine($"Predicted label: {prediction.PredictedLabel}");
    }

    public class TextData
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }

    public class TextPrediction
    {
        public string Text { get; set; }
        public string PredictedLabel { get; set; }
    }
}

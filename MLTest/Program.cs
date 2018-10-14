using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Runtime.Api;

namespace MLTest
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "DetoxData.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "DetoxTest.tsv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static async Task Main(string[] args)
        {
            var model = await Train();

            Evaluate(model);
            Predict(model);
            await model.WriteAsync("e:\\temp\\model.zip");
            Console.ReadKey();
        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 50, NumTrees = 50, MinDocumentsInLeafs = 20 });
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            await model.WriteAsync(_modelpath);
            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        public static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
                ,
                new SentimentData
                {
                    SentimentText = "Haha your mother."
                }
                                ,
                new SentimentData
                {
                    SentimentText = "Nice work."
                }
            };
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Negative" : "Positive")}");
            }
            Console.WriteLine();
        }
    }
}

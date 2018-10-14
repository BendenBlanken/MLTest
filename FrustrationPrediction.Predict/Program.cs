using Microsoft.ML.Legacy;
using Microsoft.WindowsAzure.Storage;
using Microsoft.WindowsAzure.Storage.Blob;
using MLTest;
using System;
using System.Configuration;
using System.Threading.Tasks;

namespace FrustrationPrediction.Predict
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            string storageConnectionString = ConfigurationManager.AppSettings.Get("blobStorage");
            // Check whether the connection string can be parsed.
            if (CloudStorageAccount.TryParse(storageConnectionString, out CloudStorageAccount storageAccount))
            {
            }
            else
            {
                throw new Exception("Storage connection string error.");
            }

            CloudBlobClient blobClient = storageAccount.CreateCloudBlobClient();

            //Setup our container we are going to use and create it.
            CloudBlobContainer container = blobClient.GetContainerReference("frustrationprediction");
            CloudAppendBlob appBlob = container.GetAppendBlobReference("frustrationSentiment.csv");

            var exists = await appBlob.ExistsAsync();
            if (!exists)
            {
                throw new Exception("File not found on storage account.");
            }

            //await appBlob.AppendTextAsync("\n1, Doe het nou..");

            var test = await appBlob.DownloadTextAsync();

            Console.WriteLine();
            Console.WriteLine("Frustratie voorspelling");
            Console.WriteLine("---------------------");
            Console.WriteLine("Laden van het model");
            Console.WriteLine("---------------------");
            var model = await PredictionModel.ReadAsync<SentimentData, SentimentPrediction>("e:\\temp\\model.zip");
            while (true)
            {
                Console.WriteLine("Voer je test string in:");
                var sentiment = new SentimentData
                {
                    SentimentText = Console.ReadLine()
                };

                var prediction = model.Predict(sentiment);

                bool answered = true;
                while (answered)
                {
                    Console.WriteLine($"De zin: {sentiment.SentimentText} {(prediction.Sentiment ? "bevat aanwijzingen van frustratie!" : "bevat geen frustratie!")}");
                    Console.WriteLine("Was dit juist? [J]/[N]");
                    var key = Console.ReadKey();
                    Console.WriteLine("");
                    if (key.Key.ToString().ToLower() == "j")
                    {
                        Console.WriteLine("Top! We nemen dit mee voor het trainen van het model!");
                        answered = false;
                    }
                    else if (key.Key.ToString().ToLower() == "n")
                    {
                        Console.WriteLine("Jammer. We nemen je verbetering mee voor het trainen van het model!");
                        answered = false;
                    }
                }
            }
        }
    }
}

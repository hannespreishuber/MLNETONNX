
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using MLNETONNX;


string assetsPath = @"C:\\stateof\\MLNETONNX\\MLNETONNX\\assets";
var modelFilePath = Path.Combine(assetsPath, "model.onnx");

var mlContext = new MLContext();

//var imageData = new ImageData
//{
//    Image = File.ReadAllBytes(@"c:\temp\test.png"),
//    ImagePath = @"c:\temp\test.png"
//};


//B
var imageData = new ImageData
{
    Image = MLImage.CreateFromFile(@"c:\temp\test.png")
  
};
var dataView = mlContext.Data.LoadFromEnumerable(new List<ImageData>() );

//var pipeline =
//    mlContext.Transforms.LoadImages(outputColumnName: "data", imageFolder: "", inputColumnName: nameof(ImageData.ImagePath))
//    .Append(mlContext.Transforms.ResizeImages(
//                   inputColumnName: "data",
//                   outputColumnName: "data",
//                   imageWidth: 416,
//                   imageHeight: 416,
//                   resizing: ImageResizingEstimator.ResizingKind.Fill))
//    .Append(mlContext.Transforms.ExtractPixels(inputColumnName:"data",
//                   outputColumnName: "data"))
//    .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelFilePath,
//    outputColumnNames: new[] { "model_outputs0" }, 
//      inputColumnNames: new[] { "data" }));

//B
var pipeline =  mlContext.Transforms.ResizeImages("data", 416,416, nameof(ImageData.Image))
        .Append(mlContext.Transforms.ExtractPixels( "data",  "data"))
        .Append(mlContext.Transforms.ApplyOnnxModel("model_outputs0", "data", modelFilePath));

var model = pipeline.Fit(dataView);

var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, List<PredictionModel>>(model);
var prediction = predictionEngine.Predict(imageData);

Console.WriteLine($"Vorhersage: ");



public sealed class BoundingBox
{
    public BoundingBox(float left, float top, float width, float height)
    {
        this.Left = left;
        this.Top = top;
        this.Width = width;
        this.Height = height;
    }

    public float Left { get; private set; }
    public float Top { get; private set; }
    public float Width { get; private set; }
    public float Height { get; private set; }
}

public sealed class PredictionModel
{
    public PredictionModel(float[] probability, string tagName, BoundingBox boundingBox)
    {
        this.Probability = probability;
        this.TagName = tagName;
        this.BoundingBox = boundingBox;
    }
   
    public float[] Probability { get; private set; }
    public string TagName { get; private set; }
    public BoundingBox BoundingBox { get; private set; }
}

//public class ImageData
//{
//    [ImageType(416, 416)]
//    [ColumnName("data")]
//    public byte[] Image { get; set; }
//    public string ImagePath { get; set; }
    
//}


//B
public class ImageData
{
    [ImageType(416, 416)]
    public MLImage Image { get; set; }


}

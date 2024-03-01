using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNETONNX
{
    
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
        public PredictionModel(float probability, string tagName, BoundingBox boundingBox)
        {
            this.Probability = probability;
            this.TagName = tagName;
            this.BoundingBox = boundingBox;
        }

        public float Probability { get; private set; }
        public string TagName { get; private set; }
        public BoundingBox BoundingBox { get; private set; }
    }

}

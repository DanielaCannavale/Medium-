using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class sampleCodeSnippet : MonoBehaviour
{
    public NNModel onnxAsset;
    public Texture2D imageToRecognise;
    private IWorker worker;
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Star function called.");
        if (onnxAsset != null)
        {
            Model onnxModel = ModelLoader.Load(onnxAsset);
            worker = onnxAsset.CreateWorker();
            using (var input = new Tensor(imageToRecognise, channels: 3))
            {
                var output = worker.Execute(input).PeekOutput();
                var indexWithHighestProbability = output[0];
                UnityEngine.Debug.Log($"Image was recognised ad class number: " + output[0] + " " + output[1]);
            }
        }
       
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

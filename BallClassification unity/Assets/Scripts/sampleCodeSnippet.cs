using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using TMPro;

public class sampleCodeSnippet : MonoBehaviour
{
    public NNModel onnxAsset;
    public Texture2D imageToRecognise_beach;
    public Texture2D imageToRecognise_football;
    private IWorker worker;
    public TextMeshProUGUI beachText;
    public TextMeshProUGUI footballText;


    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Star function called.");
        if (onnxAsset != null)
        {
            Model onnxModel = ModelLoader.Load(onnxAsset);
            worker = onnxAsset.CreateWorker();
            using (var input = new Tensor(imageToRecognise_beach, channels: 3))
            {
                var output = worker.Execute(input).PeekOutput();
                var indexWithHighestProbability = output[0];
                UnityEngine.Debug.Log($"Image was recognised as class number: " + output[0] + " " + output[1]);
                if (beachText != null)
                {
                    beachText.text = $"Image was recognised as class number: B: {output[0]}, F: {output[1]}";
                }
                else
                {
                    Debug.LogError("oggetto nullo");
                }
            }
            using (var input1 = new Tensor(imageToRecognise_football, channels: 3))
                {
                    var output1 = worker.Execute(input1).PeekOutput();
                    var indexWithHighestProbability = output1[0];
                    UnityEngine.Debug.Log($"Image was recognised as class number: " + output1[0] + "" + output1[1]);
                    if (footballText != null)
                    {

                        footballText.text = $"Image was recognised as class number: B: {output1[0]}, F: {output1[1]}";
                    }
                    else
                {
                    Debug.LogError("oggetto nullo");
                }
            }
        }
       
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

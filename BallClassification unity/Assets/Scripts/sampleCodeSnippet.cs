using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using TMPro;

public class sampleCodeSnippet : MonoBehaviour
{
    public NNModel onnxAsset; //definisce la variabile per il modello;
    public Texture2D imageToRecognise_beach; //definisce la prima immagine su cui si fa la predizione
    public Texture2D imageToRecognise_football; //seconda immagine su cui si fa la predizione
    private IWorker worker; //definisce il worker
    public TextMeshProUGUI beachText; //definisce le variabili per il testo
    public TextMeshProUGUI footballText;


    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Star function called.");
        if (onnxAsset != null)
        {
            Model onnxModel = ModelLoader.Load(onnxAsset); //carica il modello
            worker = onnxAsset.CreateWorker(); //crea il worker
            using (var input = new Tensor(imageToRecognise_beach, channels: 3)) //tensorizza l'input
            {
                var output = worker.Execute(input).PeekOutput(); //definisce la variabile di output
                var indexWithHighestProbability = output[0];
                UnityEngine.Debug.Log($"Image was recognised as class number: " + output[0] + " " + output[1]);
                /*if (beachText != null)
                 {
                     beachText.text = $"Image was recognised as class number: B: {output[0]}, F: {output[1]}";
                 }
                */

                if (output[0] > output[1])
                {
                    beachText.text = $" B: {output[0]}, F: {output[1]} " + $" Recognised class: Beachball";
                } 
                 
                  else if (output[1] > output[0])
                    {
                        beachText.text = $" B: {output[0]}, F: {output[1]} " + $" Recognised class: Football";
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
                    /* if (footballText != null)
                     {

                         footballText.text = $"Image was recognised as class number: B: {output1[0]}, F: {output1[1]}";
                     }*/

                    if (output1[1] > output1[0])
                    {
                        footballText.text = $" B: {output1[0]}, F: {output1[1]} " + $" Recognised class: Football";
                    }
                    else if (output1[0] > output1[1])
                        {
                            footballText.text = $" B: {output1[0]}, F: {output1[1]} " + $" Recognised class: Beachball";
                        }
                        else
                        {
                            Debug.LogError("oggetto nullo");
                        }
                    }
                }

         

            // Update is called once per frame
            void Update()
            {

            }

        }
    }


using System;
using KS.Foundation;
using Microsoft.ML.Tokenizers;
using TheBrain.CudaSNN;
using System.Text;
using System.Diagnostics;

namespace TheBrain;

public class BookTrainer
{
    private SnnModel _model;    

    SnnTokenizer Tokenizer;

    int NumInputNeurons;

    public BookTrainer(SnnModel model, int numInputNeurons)
    {
        _model = model;
        NumInputNeurons = numInputNeurons;
        Tokenizer = new SnnTokenizer(numInputNeurons);
    }

    public void TrainOnDirectory(string path)
    {
        var files = Directory.GetFiles(path, "*.txt");
        files.Shuffle();

        int tokenCount = 0;
        int stepCount = 0;

        int neuronCount = _model.NeuronCount;

        while (true)
        {
            foreach (var file in files)
            {
                foreach (var line in File.ReadLines(file))
                {
                    int[] tokenNeuronIds = Tokenizer.EncodeAndMap(line);
                    
                    for (int i = 0; i < tokenNeuronIds.Length - 1; i++)                    
                    {
                        int tokenID = tokenNeuronIds[i];
                        int nextTokenID = tokenNeuronIds[i + 1];
                        // 1. Impuls geben
                        _model.FireNeuron(tokenID);
                        _model.FireNeuron(neuronCount - NumInputNeurons + nextTokenID);
                        tokenCount++;
                        
                        // 2. Propagations-Phase
                        // Wir simulieren hier z.B. 10 Steps pro Token, 
                        // damit die Welle Zeit hat, sich auszubreiten.
                        for (int k = 0; k < 1; k++) 
                        {
                            _model.Step(1f, 1);
                            Thread.Sleep(1);
                            stepCount++;
                        }

                        /***
                        //float avgWeight = _model.GetAverageSynapseWeight();
                        var test = _model.GetNeuronPotentialsForDrawing();                        
                        int Fire = test.Count(t => t.State == 1);
                        int autoFire = test.Count(t => t.State == 2);
                        int hasPotential = test.Count(t => t.Output >= 0.1);
                        int hasDebug = test.Count(t => t.Debug == 13);

                        Debug.Assert(hasDebug == 0);

                        int synapseCount = _model.GetDynamicSynapseCount();
                        
                        //Debug.WriteLine($"Tokens: {tokenCount}, Step: {stepCount}, Fire: {Fire}, AutoFire: {autoFire}, HasPotential: {hasPotential}, Synapses: {synapseCount}, Avg-Weight: {avgWeight}, hasDebug: {hasDebug}");
                        //Debug.WriteLine($"Tokens: {tokenCount}, Step: {stepCount}, Fire: {Fire}, AutoFire: {autoFire}, HasPotential: {hasPotential}, Synapses: {synapseCount}, Avg-Weight: {avgWeight}");
                        Debug.WriteLine($"Tokens: {tokenCount}, Step: {stepCount}, Fire: {Fire}, AutoFire: {autoFire}, HasPotential: {hasPotential}, Synapses: {synapseCount}");
                        ***/
                        
                    }
                }
            }
        }
    }
}
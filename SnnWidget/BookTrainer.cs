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

    public BookTrainer(SnnModel model, int numInputNeurons)
    {
        _model = model;
        Tokenizer = new SnnTokenizer(numInputNeurons);
    }

    public void TrainOnDirectory(string path)
    {
        var files = Directory.GetFiles(path, "*.txt");
        files.Shuffle();

        int tokenCount = 0;
        int stepCount = 0;        

        while (true)
        {
            foreach (var file in files)
            {
                foreach (var line in File.ReadLines(file))
                {
                    int[] tokenNeuronIds = Tokenizer.EncodeAndMap(line);
                    
                    foreach (var neuronId in tokenNeuronIds)
                    {
                        // 1. Impuls geben
                        _model.FireNeuron(neuronId);
                        tokenCount++;
                        
                        // 2. Propagations-Phase
                        // Wir simulieren hier z.B. 10 Steps pro Token, 
                        // damit die Welle Zeit hat, sich auszubreiten.
                        for (int i = 0; i < 1; i++) 
                        {
                            _model.Step(1f, 1);
                            stepCount++;
                        }

                        /*** ***/
                        //float avgWeight = _model.GetAverageSynapseWeight();
                        var test = _model.GetNeuronPotentialsForDrawing();                        
                        int Fire = test.Count(t => t.State == 1);
                        int autoFire = test.Count(t => t.State == 2);
                        int hasPotential = test.Count(t => t.Potential >= 0.1);
                        int hasDebug = test.Count(t => t.Debug == 13);

                        Debug.Assert(hasDebug == 0);

                        int synapseCount = _model.GetDynamicSynapseCount();
                        
                        //Debug.WriteLine($"Tokens: {tokenCount}, Step: {stepCount}, Fire: {Fire}, AutoFire: {autoFire}, HasPotential: {hasPotential}, Synapses: {synapseCount}, Avg-Weight: {avgWeight}, hasDebug: {hasDebug}");
                        //Debug.WriteLine($"Tokens: {tokenCount}, Step: {stepCount}, Fire: {Fire}, AutoFire: {autoFire}, HasPotential: {hasPotential}, Synapses: {synapseCount}, Avg-Weight: {avgWeight}");
                        Debug.WriteLine($"Tokens: {tokenCount}, Step: {stepCount}, Fire: {Fire}, AutoFire: {autoFire}, HasPotential: {hasPotential}, Synapses: {synapseCount}");
                        
                    }
                }
            }
        }
    }
}
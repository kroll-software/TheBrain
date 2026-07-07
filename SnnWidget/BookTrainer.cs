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

    public SnnTokenizer Tokenizer {get; private set; }

    public event EventHandler ProgressChanged;
    
    private int m_Progress;
    public int Progress
    {
        get
        {
            return m_Progress;
        }
        set
        {
            if (m_Progress != value)
            {
                m_Progress = value;
                if (ProgressChanged != null)
                    ProgressChanged(this, EventArgs.Empty);
            }
        }
    }


    public string BookFile { get; private set; }

    public BookTrainer(SnnModel model)
    {
        _model = model;        
        Tokenizer = new SnnTokenizer();        
    }

    public void TrainOnDirectory(string path)
    {
        var files = Directory.GetFiles(path, "*.txt");
        files.Shuffle();

        int tokenCount = 0;
        int stepCount = 0;

        int neuronCount = _model.Neurons.NeuronCount;
        int delaySteps = 8;

        while (true)
        {
            foreach (var file in files)
            {
                BookFile = file;

                var lines = File.ReadLines(file);
                var lineCount = lines.Count();
                int currentLine = 0;

                foreach (var line in lines)
                {
                    currentLine++;
                    Progress = (int)((float)currentLine / lineCount  * 1000f);

                    int[] tokenNeuronIds = Tokenizer.EncodeAndMap(line);
                    
                    for (int i = 0; i < tokenNeuronIds.Length - 1; i++)                    
                    {
                        int tokenID = tokenNeuronIds[i];
                        int nextTokenID = tokenNeuronIds[i + 1];
                        // 1. Impuls geben
                        // Input
                        _model.FireNeuron(tokenID);

                        for (int k = 0; k < delaySteps; k++)
                        {
                            _model.Step();
                            Thread.Sleep(1);
                            stepCount++;
                        }

                        // Target
                        _model.FireNeuron(neuronCount - Tokenizer.VocabSize + nextTokenID);

                        _model.Step();
                        Thread.Sleep(1);
                        stepCount++;

                        tokenCount++;

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
                
                _model.SaveModel();
            }
        }
    }
}
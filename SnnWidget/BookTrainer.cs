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

        int neuronCount = _model.NeuronCount;

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
                    Progress = (int)((float)currentLine / lineCount  * 100f);

                    int[] tokenNeuronIds = Tokenizer.EncodeAndMap(line);
                    
                    for (int i = 0; i < tokenNeuronIds.Length - 1; i++)                    
                    {
                        int tokenID = tokenNeuronIds[i];
                        int nextTokenID = tokenNeuronIds[i + 1];
                        // 1. Impuls geben
                        _model.FireNeuron(tokenID);
                        _model.FireNeuron(neuronCount - Tokenizer.VocabSize + nextTokenID);
                        tokenCount++;
                        
                        // 2. Propagations-Phase
                        // Wir simulieren hier z.B. 10 Steps pro Token, 
                        // damit die Welle Zeit hat, sich auszubreiten.
                        for (int k = 0; k < 1; k++) 
                        {                            
                            //_model.Step(1f, 1);
                            _model.Step();  // use default parameters
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
                
                _model.SaveModel();
            }
        }
    }
}
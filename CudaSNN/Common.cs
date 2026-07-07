using System;

namespace TheBrain.CudaSNN;


public class BrainConfiguration
{		
    public BrainConfiguration() {}
    
    public int NumInputClasses { get; set; }
    public int NumInputClassNeurons { get; set; }
    public int NumHiddenLayers { get; set; }
    public int NeuronsPerHiddenLayer { get; set; }	
    public int HiddenLayerMaxSynapses { get; set; }	
    public int NumOutputClasses { get; set; }
    public int NumOutputClassNeurons { get; set; }
    public int OutputLayerMaxSynapses { get; set; }
}

public enum BrainModes
{
    Awake,
    Dreaming,
    Generating
}


using System;
using System.Runtime.InteropServices;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;


namespace TheBrain.CudaSNN;


[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct NeuronState : IComponent
{
    public int ID;
    // Floats
    public float Input;
    public float Output;
    public float Threshold;
    public float Energy;
    public float ShortTermExcitement;
    public float LongTermExcitement;
    public float ConnectionRadius;

    // Tracking für das Selbst-Verdrahten
    public int FirstSynapseIndex; 
    public int CurrentSynapseCount; // Wie viele Synapsen haben wir schon?
    public int MaxSynapseLimit;     // Das Limit aus deiner Konfiguration
    public int NewSynapseCounter;
    // Die Rangliste der Kandidaten (festes Array für die GPU!)
    public unsafe fixed int CandidateIndices[8]; // Top 8 Favoriten
    public unsafe fixed int CandidateScores[8];  // Wie oft zusammen gefeuert
    
    // Positionen
    public float PosX, PosY, PosZ;
    public float AxonX, AxonY, AxonZ;

    // Integers / Bytes
    public int FireCycleRemaining;

    public byte Type; // 0: Excitory, 1: Inhibitory

    public byte State; // 0: Idle, 1: Fire, 2: AutoFire

    // RADIKALER WECHSEL: Keine bools mehr im Struct!
    // Wir nutzen Byte-Flags als Ersatz für bools (0 = false, 1 = true)
    public byte IsAutoFireActive;

    public int Debug;
}


[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct NeuronRenderState
{
    public byte Type;
    public byte State;
    public float Output;        
    public float PosX;
    public float PosY;
    public float PosZ;
    public int Debug;
}
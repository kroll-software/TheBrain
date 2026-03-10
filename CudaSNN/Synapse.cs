using System;
using System.Runtime.InteropServices;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace TheBrain.CudaSNN;

// Die "unorthodoxe" Synapse in der Linked List
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct SynapseData : IComponent
{
    public int SourceNeuronIdx;
    public int TargetEntityID;
    public float Weight;
    public int NextIndex;
}

[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct CandidateEntry
{    
    public int TargetNeuronID;
    public int Score;
    public int NextEntryIndex; // -1 für Ende der Liste
}

[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct VoxelEntry
{
    public int VoxelHash;
    public int NeuronIndex;
}


[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct WeightSumResult
{
    public float TotalWeight;
    public int ActiveCount;
}
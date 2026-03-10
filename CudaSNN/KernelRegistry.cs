using System;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace TheBrain.CudaSNN;

public class KernelRegistry
{
    // Dictionary zum Speichern der geladenen Kernel
    private readonly Dictionary<string, Delegate> _kernels = new();

    public KernelRegistry(Accelerator acc)
    {        
        _kernels["FireNeuron"] = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView1D<NeuronState, Stride1D.Dense>, int, float, float>(SnnKernels.FireNeuronKernel);
        
        _kernels["Step"] = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<NeuronState, Stride1D.Dense>, float, int, int>(SnnKernels.UpdateNeuronStep);

        _kernels["ExtractRenderData"] = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<NeuronState, Stride1D.Dense>, ArrayView1D<NeuronRenderState, Stride1D.Dense>>(SnnKernels.ExtractRenderDataKernel);
        
        _kernels["GetActiveSynapseCount"] = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<SynapseData, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(SnnKernels.CountActiveSynapsesKernel);

        _kernels["HebbianSpatialKernel"] = acc.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<NeuronState, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, // watermarkBuffer
            ArrayView1D<SynapseData, Stride1D.Dense>, // synapsePool
            int, float, float>(SnnKernels.HebbianSpatialKernel);            

        _kernels["UpdateNeuronStep"] = acc.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<NeuronState, Stride1D.Dense>, 
            float, 
            int, int>(SnnKernels.UpdateNeuronStep);

        /***
        kernels["ProcessPulses"] = acc.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<NeuronState, Stride1D.Dense>, 
            ArrayView1D<int, Stride1D.Dense>, 
            ArrayView1D<SynapseData, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>(SnnKernels.ProcessPulses);            
        ***/

        _kernels["ProcessPulses"] = acc.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<NeuronState, Stride1D.Dense>,
            ArrayView1D<SynapseData, Stride1D.Dense>>(SnnKernels.ProcessPulses);            

        _kernels["CalculateWeightStatsKernel"] = acc.LoadAutoGroupedStreamKernel<
            Index1D, 
        ArrayView1D<SynapseData, Stride1D.Dense>,
        int,
        ArrayView1D<WeightSumResult, Stride1D.Dense>>(SnnKernels.CalculateWeightStatsKernel);
        
    }
    

    // Typsicherer Zugriff
    public T GetKernel<T>(string name) where T : Delegate
    {
        return (T)_kernels[name];
    }
}
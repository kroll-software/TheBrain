using System;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace TheBrain.CudaSNN;

public class RenderingSystem : GpuSystem
{
    //private GpuEngine _engine;
    public MemoryBuffer1D<NeuronRenderState, Stride1D.Dense> DeviceBuffer { get; private set; }

    public NeuronRenderState[] RenderBuffer {get; private set; } = null;

    public RenderingSystem()
    {
        //_engine = new GpuEngine();
    }

    public void AllocateDeviceMemory(int neuronCount, Accelerator accelerator)
    {
        RenderBuffer = new NeuronRenderState[neuronCount];
        DeviceBuffer = accelerator.Allocate1D<NeuronRenderState>(neuronCount);
    }

    // Member-Variablen    

    public override void UpdateGpu(double elapsedMs)
    {
        throw new NotImplementedException();
    }

    protected override void AddKeyComponents()
    {
        this.AddKeyComponent<NeuronRenderState>();
    }

    protected override bool IsDraw()
    {
        return true;
    }
}

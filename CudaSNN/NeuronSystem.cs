using System;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;


namespace TheBrain.CudaSNN;

public class NeuronSystem : GpuSystem 
{
    private GpuEngine _engine;
    public MemoryBuffer1D<NeuronState, Stride1D.Dense> DeviceBuffer { get; private set; }
    

    public NeuronSystem()
    {
        _engine = new GpuEngine();
    }

    public void AllocateDeviceMemory(int totalNeurons, Accelerator accelerator)
    {
        // Alten Speicher freigeben, falls vorhanden
        DeviceBuffer?.Dispose();
        DeviceBuffer = accelerator.Allocate1D<NeuronState>(totalNeurons);
    }

    public override void Update(IEnumerable<IEntity> entities, double elapsedMs)
    {
        throw new NotImplementedException();
    }

    public override void UpdateGpu(double elapsedMs)
    {        
    }

    protected override void AddKeyComponents()
    {
        this.AddKeyComponent<NeuronState>();
    }

    protected override bool IsDraw()
    {
        return false;
    }
}

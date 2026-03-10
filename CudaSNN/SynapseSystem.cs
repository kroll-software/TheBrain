using System;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;


namespace TheBrain.CudaSNN;

public class SynapseSystem : GpuSystem 
{
    private GpuEngine _engine;
    public MemoryBuffer1D<SynapseData, Stride1D.Dense> DeviceBuffer { get; private set; }
    
    public void AllocateDeviceMemory(int totalSynapses, Accelerator accelerator)
    {
        // Alten Speicher freigeben, falls vorhanden
        DeviceBuffer?.Dispose();
        DeviceBuffer = accelerator.Allocate1D<SynapseData>(totalSynapses);
    }

    public SynapseSystem()
    {
        _engine = new GpuEngine();
    }

    public override void UpdateGpu(double elapsedMs)
    {
        throw new NotImplementedException();
    }

    protected override bool IsDraw()
    {
        // wir sind ein reines Rechen-System und rendern nichts auf den Bildschirm
        return false;
    }

    protected override void AddKeyComponents()
    {
        // Die Anmeldung der Kern-Komponente
        this.AddKeyComponent<SynapseData>();
    }
}
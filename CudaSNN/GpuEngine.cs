using System;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;


namespace TheBrain.CudaSNN;

public class GpuEngine : IDisposable
{
    public Context Context { get; }
    public Accelerator Accelerator { get; }

    public GpuEngine()
    {
        // Erstellt den ILGPU Kontext (automatisch für CUDA)
        Context = Context.Create(builder => builder.Cuda());
        
        // Holt sich die erste verfügbare NVIDIA-Grafikkarte
        Accelerator = Context.GetCudaDevice(0).CreateAccelerator(Context);
    }

    public void Dispose()
    {
        Accelerator?.Dispose();
        Context?.Dispose();
    }
}
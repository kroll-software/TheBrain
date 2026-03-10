using System;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace TheBrain.CudaSNN;

public abstract class GpuSystem : BaseSystem 
{
    // Wir umgehen das IEnumerable<IEntity> für die GPU
    public abstract void UpdateGpu(double elapsedMs);
}
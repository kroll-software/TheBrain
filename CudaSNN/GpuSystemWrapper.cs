using System;
using KS.Foundation.ECS;

namespace TheBrain.CudaSNN;

public class GpuSystemWrapper
{
    protected IEnumerable<GpuSystem> _systems;

    /***
    public void Update(double elapsedMs) {
        foreach(var wrapper in _systems) {
            if(wrapper.System is GpuSystem gpuSystem) {
                gpuSystem.UpdateGpu(elapsedMs);
            } else {
                wrapper.System.Update(elapsedMs);
            }
        }
    }
    ***/
}

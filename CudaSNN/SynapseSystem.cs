using System;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;


namespace TheBrain.CudaSNN;

public class SynapseSystem : GpuSystem 
{
    public int SynapseCount {get; private set; } = 0;

    public MemoryBuffer1D<int, Stride1D.Dense> WatermarkBuffer {get; private set; }

    //public MemoryBuffer1D<SynapseData, Stride1D.Dense> synapsePool {get; private set; };
    public long SynapsePoolCapacity => DeviceBuffer.Length;    

    //private GpuEngine _engine;
    public MemoryBuffer1D<SynapseData, Stride1D.Dense> DeviceBuffer { get; private set; }
    
    public void AllocateDeviceMemory(IWorld world, int maxTotalSynapses, Accelerator accelerator)
    {
        // Alten Speicher freigeben, falls vorhanden
        DeviceBuffer?.Dispose();
        WatermarkBuffer?.Dispose();

        DeviceBuffer = accelerator.Allocate1D<SynapseData>(maxTotalSynapses);

        WatermarkBuffer = accelerator.Allocate1D<int>(1);
        WatermarkBuffer.MemSet(0);

        SynapseCount = maxTotalSynapses;

        var allSynapses = new List<SynapseData>();
        world.EntityFinder.Find<SynapseData>().ForEach(entity => {
            if (entity.TryGet<SynapseData>(out var syn)) allSynapses.Add(syn);
        });

        if (allSynapses.Count > 0)
            DeviceBuffer.CopyFromCPU(allSynapses.ToArray());        

        Console.WriteLine($"{SynapseCount} Synapsen erfolgreich auf die GPU geladen.");
    }

    public int GetDynamicSynapseCount()
    {        
        int[] result = new int[1];
        WatermarkBuffer.CopyToCPU(result);
        return result[0];
    }    

    public SynapseData[] GetSynapses()
    {
        int count = GetDynamicSynapseCount();
        SynapseData[] result = new SynapseData[count];
        DeviceBuffer.View.SubView(0, count).CopyToCPU(result);
        return result;
    }

    public void SetSynapses(SynapseData[] synapses)
    {
        if (synapses.Length > DeviceBuffer.Length)
            throw new Exception(
                $"Snapshot enthält {synapses.Length} Synapsen, " +
                $"aber Poolgröße ist nur {DeviceBuffer.Length}");

        // Synapsen in den Pool kopieren
        DeviceBuffer.View.SubView(0, synapses.Length).CopyFromCPU(synapses);

        // Watermark setzen
        int[] watermark = new int[1];
        watermark[0] = synapses.Length;

        WatermarkBuffer.CopyFromCPU(watermark);
    }

    public SynapseSystem()
    {
        //_engine = new GpuEngine();
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
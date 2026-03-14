using System;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SummerGUI;


namespace TheBrain.CudaSNN;

public class NeuronSystem : GpuSystem 
{
    public int NeuronCount { get; protected set; }

    //private GpuEngine _engine;
    public MemoryBuffer1D<NeuronState, Stride1D.Dense> DeviceBuffer { get; private set; }
    
    public MemoryBuffer1D<int, Stride1D.Dense> sortedNeuronIDs { get; private set; }

    public int gridDim { get; private set; } = 32;
    public float voxelSize { get; private set; }
    public float WorldSize { get; private set; }
    public float MaxReachSq { get; private set; }
    public MemoryBuffer1D<int, Stride1D.Dense> gridLookup { get; private set; }

    public NeuronSystem()
    {
        //_engine = new GpuEngine();
    }

    public void AllocateDeviceMemory(IWorld world, Accelerator accelerator)
    {
        // Reset
        NeuronCount = 0;

        // Release memory
        DeviceBuffer?.Dispose();

        // Get Neuron Components
        var components = new List<NeuronState>();
        world.EntityFinder.Find<NeuronState>().ForEach(entity =>
        {
            if (entity.TryGet<NeuronState>(out var state))
                components.Add(state);
        });

        NeuronCount = components.Count;
        if (NeuronCount == 0) return;

        // Allocate Memory        
        DeviceBuffer = accelerator.Allocate1D<NeuronState>(NeuronCount);
        // 3. Daten hochladen        
        DeviceBuffer.CopyFromCPU(components.ToArray());
        Console.WriteLine($"{NeuronCount} Neuronen erfolgreich auf die GPU geladen.");        

        OnAfterLoad(components.ToArray(), accelerator);
    }

    public void OnAfterLoad(NeuronState[] neurons, Accelerator accelerator)
    {
        sortedNeuronIDs?.Dispose();
        gridLookup?.Dispose();

        // Create other structures
        sortedNeuronIDs = accelerator.Allocate1D<int>(NeuronCount);
        gridLookup = accelerator.Allocate1D<int>(gridDim * gridDim * gridDim + 1);
        int[] sortedNeuronIDsCPU;
        int[] gridLookupCPU;
        BuildGrid(neurons, gridDim, voxelSize,
          out sortedNeuronIDsCPU,
          out gridLookupCPU);
        sortedNeuronIDs.CopyFromCPU(sortedNeuronIDsCPU);
        gridLookup.CopyFromCPU(gridLookupCPU);
    }

    public static void BuildGrid(
        NeuronState[] neurons,
        int gridDim,
        float voxelSize,
        out int[] sortedNeuronIDs,
        out int[] gridLookup)
    {
        int neuronCount = neurons.Length;
        int cellCount = gridDim * gridDim * gridDim;

        sortedNeuronIDs = new int[neuronCount];
        gridLookup = new int[cellCount + 1];

        // temporäre Liste von (cell, neuronID)
        var pairs = new (int cell, int id)[neuronCount];

        for (int i = 0; i < neuronCount; i++)
        {
            int cx = (int)(neurons[i].AxonX / voxelSize);
            int cy = (int)(neurons[i].AxonY / voxelSize);
            int cz = (int)(neurons[i].AxonZ / voxelSize);

            cx = Math.Clamp(cx, 0, gridDim - 1);
            cy = Math.Clamp(cy, 0, gridDim - 1);
            cz = Math.Clamp(cz, 0, gridDim - 1);

            int cell = cx + gridDim * cy + gridDim * gridDim * cz;

            pairs[i] = (cell, i);
        }

        // Nach Zellindex sortieren
        Array.Sort(pairs, (a, b) => a.cell.CompareTo(b.cell));

        // sortedNeuronIDs füllen
        for (int i = 0; i < neuronCount; i++)
            sortedNeuronIDs[i] = pairs[i].id;

        // gridLookup bauen
        int currentCell = 0;

        for (int i = 0; i < neuronCount; i++)
        {
            int cell = pairs[i].cell;

            while (currentCell <= cell)
            {
                gridLookup[currentCell] = i;
                currentCell++;
            }
        }

        while (currentCell <= cellCount)
        {
            gridLookup[currentCell] = neuronCount;
            currentCell++;
        }
    }

    public void AutoConfigureMetrics(int gridDim, float fixedVoxelSize)
    {
        // Die Welt ist exakt so groß wie das Grid
        this.voxelSize = fixedVoxelSize;
        this.WorldSize = gridDim * voxelSize; 
        
        // maxReach bleibt eine Funktion der Voxelgröße
        this.MaxReachSq = (float)Math.Pow(voxelSize * 1.8f, 2);
        //this.MaxReachSq = MathF.Pow(voxelSize * 0.66667f, 2);
        //this.MaxReachSq = MathF.Pow(voxelSize, 2);
    }    

    private int NextPowerOfTwo(int n)
    {
        int k = 1;
        while (k < n) k *= 2;
        return k;
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

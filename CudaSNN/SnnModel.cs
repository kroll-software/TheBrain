using System;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using KS.Foundation;
using KS.Foundation.ECS;


namespace TheBrain.CudaSNN;

/// <summary>
/// SnnModel liefert HighLevel API-Funktionen für alle SNN-Funktionen
/// </summary>
public class SnnModel : DisposableObject
{
    public IWorld World { get; private set; }
    public GpuEngine Gpu { get; private set; } // Die Brücke zur GPU

    public string Name { get; private set; }    

    public NeuronSystem Neurons { get; private set; }
    public SynapseSystem Synapses { get; private set; } 

    private readonly KernelRegistry m_KernelRegistry;

    public SnnModel(string name)
    {
        Name = name;
        Gpu = new GpuEngine(); // GPU-Kontext initialisieren

        m_KernelRegistry = new KernelRegistry(Gpu.Accelerator);

        Neurons = new NeuronSystem();
        Synapses = new SynapseSystem();

        BaseSystem[] systems = [
            Neurons,
            Synapses,
        ];

        World = WorldFactory.Create(systems);
    }

    public void ConnectLayers(int sourceOffset, int sourceCount, int targetOffset, int targetCount, float probability, float defaultWeight)
    {
        Random rand = new Random();
        for (int i = 0; i < sourceCount; i++)
        {
            int sourceGlobalIdx = sourceOffset + i;
            
            for (int j = 0; j < targetCount; j++)
            {
                if (rand.NextDouble() < probability)
                {
                    int targetGlobalIdx = targetOffset + j;
                    
                    // Hier fügen wir die Synapse in unseren CPU-Spiegel des Pools ein
                    // Und verknüpfen sie in der Linked List des Quell-Neurons
                    AddSynapseToPool(sourceGlobalIdx, targetGlobalIdx, defaultWeight);
                }
            }
        }
    }

    public void AddSynapseToPool(int sourceGlobalIdx, int targetGlobalIdx, float defaultWeight = 0.1f)
    {
    }

    int currentGlobalCount = 0;

    public void AddLayer(int neuronCount)
    {
        // Wir registrieren die Neuronen in der World
        // Und merken uns den Start-Index für die GPU
        int startIdx = currentGlobalCount;
        for(int i = 0; i < neuronCount; i++) {
            var entity = World.CreateEntity();
            // World.AddComponent(entity, new NeuronState { ... });
        }
        currentGlobalCount += neuronCount;
    }

    public int CalculateMaxTotalSynapses(BrainConfiguration config)
    {
        // 1. Input Layer: (Anzahl Klassen * Neuronen pro Klasse * Synapsen pro Neuron)
        int inputSynapses = config.NumInputClasses * config.NumInputClassNeurons * config.OutputLayerMaxSynapses;

        // 2. Hidden Layers: (Anzahl Layer * Neuronen pro Layer * HiddenLayerMaxSynapses)
        int hiddenSynapses = config.NumHiddenLayers * config.NeuronsPerHiddenLayer * config.HiddenLayerMaxSynapses;

        // 3. Output Layer: (Anzahl Klassen * Neuronen pro Klasse * Synapsen pro Neuron)
        int outputSynapses = config.NumOutputClasses * config.NumOutputClassNeurons * config.OutputLayerMaxSynapses;

        // Sicherheits-Puffer von 20% für dynamisches Wachstum
        return (int)((inputSynapses + hiddenSynapses + outputSynapses) * 1.2f);
    }

    public void BuildNetwork(BrainConfiguration config)
    {        
        NeuronCount = config.NumInputClasses * config.NumInputClassNeurons
            + config.NumHiddenLayers * config.NeuronsPerHiddenLayer
            + config.NumOutputClasses * config.NumOutputClassNeurons;        

        int gridDim = 64; 
        float voxelSize = 10.0f; // Ein Neuron pro 10 Einheiten
    
        AutoConfigureMetrics(gridDim, voxelSize);        
        AutoConfigureMetrics(32, 20f);

        List<int> layerOffsets = new List<int>();
        int currentOffset = 0;
        int layerIndex = 0; // Wir nutzen diesen als globalen Layer-Zähler

        // --- INPUT LAYER ---
        if (config.NumInputClasses * config.NumInputClassNeurons > 0)
        {
            layerOffsets.Add(currentOffset);
            CreateLayer(config.NumInputClasses * config.NumInputClassNeurons, currentOffset, layerType: "Input", layerIndex: layerIndex++, config);
            currentOffset += config.NumInputClasses * config.NumInputClassNeurons;
        }

        // --- HIDDEN LAYERS ---
        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            layerOffsets.Add(currentOffset);
            CreateLayer(config.NeuronsPerHiddenLayer, currentOffset, layerType: "Hidden", layerIndex: layerIndex++, config);
            currentOffset += config.NeuronsPerHiddenLayer;
        }

        // --- OUTPUT LAYER ---
        if (config.NumOutputClasses * config.NumOutputClassNeurons > 0)
        {
            layerOffsets.Add(currentOffset);
            // KORREKTUR: LayerType muss "Output" sein
            CreateLayer(config.NumOutputClasses * config.NumOutputClassNeurons, currentOffset, layerType: "Output", layerIndex: layerIndex++, config);
            currentOffset += config.NumOutputClasses * config.NumOutputClassNeurons;
        }

        UploadToGpu(config);
    }

    public int NeuronCount { get; protected set; }
    public int SynapseCount { get; protected set; }

    public MemoryBuffer1D<float, Stride1D.Dense> GlobalPotentials { get; protected set; }
    public MemoryBuffer1D<int, Stride1D.Dense> FirstSynapseBuffer { get; protected set; }

    public void UploadToGpu(BrainConfiguration config)
    {
        NeuronCount = 0;
        SynapseCount = 0;

        // *** Neuronen

        // 1. Liste in Array umwandeln (wichtig für den Pointer-Zugriff von ILGPU)
        var components = new List<NeuronState>();
        World.EntityFinder.Find<NeuronState>().ForEach(entity =>
        {
            if (entity.TryGet<NeuronState>(out var state))
                components.Add(state);
        });

        NeuronCount = components.Count;
        this._renderBuffer = new NeuronRenderState[NeuronCount];
        this._renderDeviceBuffer = this.Gpu.Accelerator.Allocate1D<NeuronRenderState>(NeuronCount);
        _synapseWatermarkBuffer = Gpu.Accelerator.Allocate1D<int>(1);
        _synapseWatermarkBuffer.MemSet(0);

        int maxTotalSynapses = CalculateMaxTotalSynapses(config);
        synapsePool = Gpu.Accelerator.Allocate1D<SynapseData>(maxTotalSynapses);        

        sortedNeuronIDs = Gpu.Accelerator.Allocate1D<int>(NeuronCount);
        gridLookup = Gpu.Accelerator.Allocate1D<int>(gridDim * gridDim * gridDim + 1);
        int[] sortedNeuronIDsCPU;
        int[] gridLookupCPU;
        BuildGrid(components.ToArray(), gridDim, voxelSize,
          out sortedNeuronIDsCPU,
          out gridLookupCPU);
        sortedNeuronIDs.CopyFromCPU(sortedNeuronIDsCPU);
        gridLookup.CopyFromCPU(gridLookupCPU);

        if (NeuronCount == 0) return;
        
        var componentArray = components.ToArray();        

        // 2. GPU Speicher im System vorbereiten
        Neurons.AllocateDeviceMemory(componentArray.Length, Gpu.Accelerator);

        // 3. Daten hochladen        
        Neurons.DeviceBuffer.CopyFromCPU(componentArray);

        // 4. Synchronisieren
        // Gpu.Accelerator.Synchronize(); ganz zuletzt
        
        Console.WriteLine($"{componentArray.Length} Neuronen erfolgreich auf die GPU geladen.");


        // *** Synapsen
        
        // 1. Buffer allokieren und mit -1 initialisieren
        
        FirstSynapseBuffer = Gpu.Accelerator.Allocate1D<int>(NeuronCount);        
        int[] initArray = new int[NeuronCount];
        Array.Fill(initArray, -1); // Füllt das CPU-Array mit -1
        FirstSynapseBuffer.CopyFromCPU(initArray); // Ein einzelner Upload-Befehl

        // 2. Synapsen Daten sammeln und hochladen
        var allSynapses = new List<SynapseData>();
        World.EntityFinder.Find<SynapseData>().ForEach(entity => {
            if (entity.TryGet<SynapseData>(out var syn)) allSynapses.Add(syn);
        });

        SynapseCount = allSynapses.Count;
        // Initialisierung (wichtig: getrennte Zuweisung!)                
        
        Synapses.AllocateDeviceMemory(Math.Max(1, SynapseCount), Gpu.Accelerator);
        if (SynapseCount > 0)
        {        
            Synapses.DeviceBuffer.CopyFromCPU(allSynapses.ToArray());
            
            // HIER MÜSSTE NOCH LOGIK REIN: 
            // Berechne hier, welches Neuron welche Synapse besitzt 
            // und schreibe das in den FirstSynapseBuffer (via CopyFromCPU)
        }
        
        // 3. Globale Potenziale
        GlobalPotentials = Gpu.Accelerator.Allocate1D<float>(NeuronCount);
        // Sicherstellen, dass sie zu Beginn auf 0 stehen
        GlobalPotentials.MemSetToZero();

        Console.WriteLine($"{SynapseCount} Synapsen erfolgreich auf die GPU geladen.");

        // *** Synchronisieren
        Gpu.Accelerator.Synchronize();        
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
            int cx = (int)(neurons[i].PosX / voxelSize);
            int cy = (int)(neurons[i].PosY / voxelSize);
            int cz = (int)(neurons[i].PosZ / voxelSize);

            // Clamp für Sicherheit
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

    private void CreateLayer(int count, int startIdx, string layerType, int layerIndex, BrainConfiguration config)
    {
        Random rand = new Random();

        int maxSynapses = 0;
        switch (layerType)
        {
            case "Input":
                maxSynapses = 0;
                break;

            case "Output":
                maxSynapses = config.OutputLayerMaxSynapses;
                break;

            case "Hidden":
                maxSynapses = config.HiddenLayerMaxSynapses;
                break;
        }
        
        // Einfache, gleichmäßige Verteilung im gesamten Würfel [0, WorldSize]
        for (int i = 0; i < count; i++)
        {
            var entity = World.CreateEntity(Neurons);            

            var state = new NeuronState
            {
                Threshold = 1,
                Energy = 1,
                ConnectionRadius = 15.0f, 
                IsAutoFireActive = (layerType == "Hidden") ? (byte)1 : (byte)0,
                MaxSynapseLimit = maxSynapses,

                PosX = (float)rand.NextDouble() * WorldSize,
                PosY = (float)rand.NextDouble() * WorldSize,
                PosZ = (float)rand.NextDouble() * WorldSize,
                
                AxonX = (float)rand.NextDouble() * WorldSize,
                AxonY = (float)rand.NextDouble() * WorldSize,
                AxonZ = (float)rand.NextDouble() * WorldSize,                
                
                // Rest der Initialisierung...
                FirstSynapseIndex = -1,
                CurrentSynapseCount = 0
            };            
            World.AddComponents(entity, state);
        }
    }

    private int _globalSeed = 0;

    public void Step(float energyRecovery, int fireCycleDuration)
    {
        _globalSeed++;
        
        // 1. UpdateNeuronStep: Alle Neuronen verarbeiten
        // Wir starten den Kernel und legen ihn in den Stream
        var updateKernel = m_KernelRegistry.GetKernel<Action<
            Index1D, 
            ArrayView1D<NeuronState, Stride1D.Dense>, 
            float, 
            int, int>>("UpdateNeuronStep");
        
        // 2. Rufe ihn auf (ohne 'stream' als Argument, das macht ILGPU automatisch, wenn du den Kernel-Typ korrekt lädst)
        // Der 'Index1D' ist für die Grid-Größe (Anzahl Neuronen)
        updateKernel(new Index1D(NeuronCount), Neurons.DeviceBuffer.View, energyRecovery, fireCycleDuration, _globalSeed);

        // 2. PropagatePulses: Hier verteilen wir Energie über die Synapsen

        var pulseKernel = m_KernelRegistry.GetKernel<Action<
            Index1D, 
            ArrayView1D<NeuronState, Stride1D.Dense>,
            ArrayView1D<SynapseData, Stride1D.Dense>>>("ProcessPulses");

        
        // Pulse werden direkt nach dem Update-Kernel in den gleichen Stream eingereiht
        pulseKernel(
            new Index1D(NeuronCount), 
            Neurons.DeviceBuffer.View, 
            synapsePool.View
        );        

        // *** Hebb: What fires together, wires together                
        var hebbianKernel = m_KernelRegistry.GetKernel<Action<
            Index1D,
            ArrayView1D<NeuronState, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>,
            ArrayView1D<int, Stride1D.Dense>, // watermarkBuffer
            ArrayView1D<SynapseData, Stride1D.Dense>, // synapsePool
            int, float, float>>("HebbianSpatialKernel");        

        hebbianKernel(
            (Index1D)Neurons.DeviceBuffer.Length,            
            Neurons.DeviceBuffer.View,
            sortedNeuronIDs.View,
            gridLookup.View,
            _synapseWatermarkBuffer.View,
            synapsePool.View,
            gridDim,
            voxelSize,
            MaxReachSq
        );
    }

    public void AutoConfigureMetrics(int gridDim, float fixedVoxelSize)
    {
        // Die Welt ist exakt so groß wie das Grid
        this.voxelSize = fixedVoxelSize;
        this.WorldSize = gridDim * voxelSize; 
        
        // maxReach bleibt eine Funktion der Voxelgröße
        this.MaxReachSq = (float)Math.Pow(voxelSize * 1.8f, 2);
    }

    private int NextPowerOfTwo(int n)
    {
        int k = 1;
        while (k < n) k *= 2;
        return k;
    }

    int gridDim = 32;
    float voxelSize;
    float WorldSize;
    float MaxReachSq;
    MemoryBuffer1D<int, Stride1D.Dense> gridLookup;


    public float GetAverageSynapseWeight()
    {        
        if (Synapses.DeviceBuffer == null)
            return 0;

        //Gpu.Accelerator.DefaultStream.Synchronize();
        var weights = Synapses.DeviceBuffer.GetAsArray1D();        

        return weights.Average(s => s.Weight);
    }

    public float GetAverageSynapseWeight_neu()
    {
        // 1. Wie viele Synapsen wurden wirklich gebildet?
        int activeCount = _synapseWatermarkBuffer.GetAsArray1D()[0];
        if (activeCount <= 0) return 0f;

        // 2. Ergebnis-Buffer auf GPU vorbereiten
        using var resBuffer = Gpu.Accelerator.Allocate1D<WeightSumResult>(1);
        resBuffer.MemSetToZero();

        var statsKernel = m_KernelRegistry.GetKernel<Action<
        Index1D, 
        ArrayView1D<SynapseData, Stride1D.Dense>,
        int,
        ArrayView1D<WeightSumResult, Stride1D.Dense>>>("CalculateWeightStatsKernel");

        // 3. Kernel starten (nur über die aktiven Synapsen)
        statsKernel(
            activeCount, 
            Synapses.DeviceBuffer.View, 
            activeCount, 
            resBuffer.View
        );
        
        // 4. Ergebnis zurückholen
        var result = resBuffer.GetAsArray1D()[0];

        if (result.ActiveCount == 0) return 0f;
        return result.TotalWeight / result.ActiveCount;
    }
    
    // In deiner SNN-Klasse
    private MemoryBuffer1D<int, Stride1D.Dense> _synapseWatermarkBuffer;

    MemoryBuffer1D<SynapseData, Stride1D.Dense> synapsePool;
    MemoryBuffer1D<int, Stride1D.Dense> sortedNeuronIDs;    

    public int GetDynamicSynapseCount()
    {        
        int[] result = new int[1];
        _synapseWatermarkBuffer.CopyToCPU(result);
        return result[0];
    }

    // Member-Variablen
    private NeuronRenderState[] _renderBuffer = null;

    private MemoryBuffer1D<NeuronRenderState, Stride1D.Dense> _renderDeviceBuffer = null;

    public NeuronRenderState[] GetNeuronPotentialsForDrawing()
    {        
        var _extractKernel = m_KernelRegistry.GetKernel<Action<Index1D, ArrayView1D<NeuronState, Stride1D.Dense>, ArrayView1D<NeuronRenderState, Stride1D.Dense>>>("ExtractRenderData");

        // 1. GPU: Daten extrahieren (dauert auf der GPU Millisekunden)
        _extractKernel(new Index1D(NeuronCount), Neurons.DeviceBuffer.View, _renderDeviceBuffer.View);
        
        _renderDeviceBuffer.View.CopyToCPU(_renderBuffer);
        
        return _renderBuffer;
    }    

    public void AddSynapse(int sourceId, int targetId)
    {
        // 1. Temporäres Array
        int[] firstSynapses = new int[NeuronCount];
        
        // 2. Nutze den View des Buffers! 
        // Diese Methode kopiert den kompletten Buffer in das Array
        FirstSynapseBuffer.View.CopyToCPU(firstSynapses);

        // 3. Logik auf der CPU...
        // (Deine Synapsen-Manipulation)
        // ToDo:
        
        // 4. Zurückschreiben
        FirstSynapseBuffer.View.CopyFromCPU(firstSynapses);
    }

    public void FireNeuron(int neuronIndex, float potential = 1f, float energy = 3f)
    {        
        var fireKernel = m_KernelRegistry.GetKernel<Action<Index1D, ArrayView1D<NeuronState, Stride1D.Dense>, int, float, float>>("FireNeuron");

        // Wir starten ihn nur für das eine spezifische Neuron (Index1D(1) reicht, 
        // aber wir übergeben den Index als Argument)
        fireKernel(new Index1D(NeuronCount), Neurons.DeviceBuffer.View, neuronIndex, potential, energy);

        //Gpu.Accelerator.Synchronize();
    }

    protected override void CleanupManagedResources()
    {
        base.CleanupManagedResources();
    }

    protected override void CleanupUnmanagedResources()
    {
        Gpu?.Dispose();
        base.CleanupUnmanagedResources();
    }
}

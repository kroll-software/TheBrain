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

    public string ModelDir { get; set; } = "/home/detlef/KSAll/OpenSource2026/TheBrain/models/";

    public IWorld World { get; private set; }
    public GpuEngine Gpu { get; private set; } // Die Brücke zur GPU

    public string Name { get; private set; }    

    public NeuronSystem Neurons { get; private set; }
    public SynapseSystem Synapses { get; private set; } 

    public float LearningRate { get; set; } = 0.005f;

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

    public void SaveModel()
    {        
        if (!Directory.Exists(ModelDir))
            Directory.CreateDirectory(ModelDir);

        string fullPath = Path.Combine(ModelDir, "model.json");

        SnnSerializer serializer = new SnnSerializer();
        serializer.Save(fullPath, this);
    }

    public void LoadModel()
    {
        string fullPath = Path.Combine(ModelDir, "model.json");
        
        SnnSerializer serializer = new SnnSerializer();
        serializer.Load(fullPath, this);
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
        Iteration = 0;

        NeuronCount = config.NumInputClasses * config.NumInputClassNeurons
            + config.NumHiddenLayers * config.NeuronsPerHiddenLayer
            + config.NumOutputClasses * config.NumOutputClassNeurons;        

        int gridDim = 32;
        float voxelSize = 20.0f; // Ein Neuron pro 10 Einheiten    
        AutoConfigureMetrics(gridDim, voxelSize);        

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

        string fullPath = Path.Combine(ModelDir, "model.json");
        
        if (!string.IsNullOrEmpty(ModelDir) && File.Exists(fullPath))
        {        
            LoadModel(); 
        }

        AllocateBuffers();
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
        if (NeuronCount == 0) return;

        this._renderBuffer = new NeuronRenderState[NeuronCount];
        this._renderDeviceBuffer = this.Gpu.Accelerator.Allocate1D<NeuronRenderState>(NeuronCount);
        _synapseWatermarkBuffer = Gpu.Accelerator.Allocate1D<int>(1);
        _synapseWatermarkBuffer.MemSet(0);

        int maxTotalSynapses = CalculateMaxTotalSynapses(config);
        synapsePool = Gpu.Accelerator.Allocate1D<SynapseData>(maxTotalSynapses);
        
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

        //Console.WriteLine($"{SynapseCount} Synapsen erfolgreich auf die GPU geladen.");

        // *** Synchronisieren
        Gpu.Accelerator.Synchronize();        
    }

    private void AllocateBuffers()
    {
        if (NeuronCount == 0)
            return;

        var components = new List<NeuronState>();
        World.EntityFinder.Find<NeuronState>().ForEach(entity =>
        {
            if (entity.TryGet<NeuronState>(out var state))
                components.Add(state);
        });

        sortedNeuronIDs = Gpu.Accelerator.Allocate1D<int>(NeuronCount);
        gridLookup = Gpu.Accelerator.Allocate1D<int>(gridDim * gridDim * gridDim + 1);
        int[] sortedNeuronIDsCPU;
        int[] gridLookupCPU;
        BuildGrid(components.ToArray(), gridDim, voxelSize,
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

    private unsafe void CreateLayer(int count, int startIdx, string layerType, int layerIndex, BrainConfiguration config)
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

        Random rnd = new Random();        
        
        // Einfache, gleichmäßige Verteilung im gesamten Würfel [0, WorldSize]
        for (int i = 0; i < count; i++)
        {
            var entity = World.CreateEntity(Neurons);            

            float posX, posY, posZ;
            float axonX = 0;
            float  axonY = 0;
            float axonZ = 0;
            int neuronsPerRow = (int)WorldSize;

            switch (layerType)
            {
                case "Input":
                    // Platzierung oben: X verteilt sich, Y ist nah bei 0
                    posX = i % neuronsPerRow; 
                    posY = (i / neuronsPerRow) * 2.0f + 1f; // Mehrere Zeilen mit 2f Abstand
                    posZ = (float)WorldSize / 2f;

                    axonX = posX;
                    axonY = (float)rand.NextDouble() * WorldSize / 2f;
                    axonZ = (float)rand.NextDouble() * WorldSize;
                    break;

                case "Output":
                    // Platzierung unten: X verteilt sich, Y ist nah bei WorldSize
                    posX = i % neuronsPerRow;
                    // Wir ziehen die Zeilen von der Unterkante nach oben ab
                    posY = WorldSize - ((i / neuronsPerRow) * 2.0f -1f); 
                    posZ = (float)WorldSize / 2f;

                    axonX = posX;
                    axonY = (WorldSize / 2f) + ((float)rand.NextDouble() * WorldSize / 2f);
                    axonZ = (float)rand.NextDouble() * WorldSize;
                    break;

                default: // Hidden
                    posX = (float)rand.NextDouble() * WorldSize;
                    posY = (float)rand.NextDouble() * WorldSize;
                    posZ = (float)rand.NextDouble() * WorldSize;
                    break;
            }

            if (axonX == 0 && axonY == 0 && axonZ == 0)
            {
                // Axon-Platzierung mit Mindestabstand            
                float minDistance = voxelSize * 1.1f; // Mindestens über die Zellgrenze hinaus
                float minDistanceSq = minDistance * minDistance;
                
                int attempts = 0;
                do
                {
                    axonX = (float)rand.NextDouble() * WorldSize;
                    axonY = (float)rand.NextDouble() * WorldSize;
                    axonZ = (float)rand.NextDouble() * WorldSize;
                    attempts++;

                    // Berechne Distanzquadrat zum Soma (Pos)
                    float dx = axonX - posX;
                    float dy = axonY - posY;
                    float dz = axonZ - posZ;
                    float distSq = dx * dx + dy * dy + dz * dz;

                    // Wenn Abstand groß genug oder wir zu viele Versuche haben (Notbremse)
                    if (distSq >= minDistanceSq || attempts > 100)
                        break;

                } while (true);
            }

            var state = new NeuronState
            {
                ID = startIdx + i,
                Threshold = 3,
                Energy = 1,
                ConnectionRadius = 5.0f, 
                CanAutoFire = (layerType == "Hidden") ? (byte)1 : (byte)0,
                MaxSynapseLimit = maxSynapses,
                Type = (layerType == "Hidden") ? (rnd.NextDouble() > 0.8) ? (byte)1 : (byte)0 : (byte)2,

                PosX = posX,
                PosY = posY,
                PosZ = posZ,
                
                AxonX = axonX,
                AxonY = axonY,
                AxonZ = axonZ,
                
                // Rest der Initialisierung...
                FirstSynapseIndex = -1,
                CurrentSynapseCount = 0
            };

            for (int k = 0; k < 15; k++)
            {
                state.CandidateIndices[k] = -1;
                state.CandidateScores[k] = 0;
            }

            World.AddComponents(entity, state);
        }
    }

    private int _globalSeed = 0;

    public int Iteration {get; set;}

    public void Step(float energyRecovery = 0.01f, int fireCycleDuration = 8)
    {
        _globalSeed++;
        Iteration++;
        
        // 1. UpdateNeuronStep: Alle Neuronen verarbeiten
        // Wir starten den Kernel und legen ihn in den Stream
        var updateKernel = m_KernelRegistry.GetKernel<Action<
            Index1D, 
            ArrayView1D<NeuronState, Stride1D.Dense>, 
            float, 
            int, int>>("UpdateNeuronStep");
        
        // 2. Rufe ihn auf (ohne 'stream' als Argument, das macht ILGPU automatisch, wenn du den Kernel-Typ korrekt lädst)
        // Der 'Index1D' ist für die Grid-Größe (Anzahl Neuronen)
        updateKernel(
            new Index1D(NeuronCount), 
            Neurons.DeviceBuffer.View, 
            energyRecovery, 
            fireCycleDuration, 
            _globalSeed
        );

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
            int, float, float, float>>("HebbianSpatialKernel");        

        hebbianKernel(
            (Index1D)Neurons.DeviceBuffer.Length,            
            Neurons.DeviceBuffer.View,
            sortedNeuronIDs.View,
            gridLookup.View,
            _synapseWatermarkBuffer.View,
            synapsePool.View,
            gridDim,
            voxelSize,
            MaxReachSq,
            LearningRate
        );
    }

    public void AutoConfigureMetrics(int gridDim, float fixedVoxelSize)
    {
        // Die Welt ist exakt so groß wie das Grid
        this.voxelSize = fixedVoxelSize;
        this.WorldSize = gridDim * voxelSize; 
        
        // maxReach bleibt eine Funktion der Voxelgröße
        //this.MaxReachSq = (float)Math.Pow(voxelSize * 1.8f, 2);
        this.MaxReachSq = MathF.Pow(voxelSize * 0.66667f, 2);
        //this.MaxReachSq = MathF.Pow(voxelSize, 2);
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
    public long SynapsePoolCapacity => synapsePool.Length;
    MemoryBuffer1D<int, Stride1D.Dense> sortedNeuronIDs;    

    public int GetDynamicSynapseCount()
    {        
        int[] result = new int[1];
        _synapseWatermarkBuffer.CopyToCPU(result);
        return result[0];
    }

    public SynapseData[] GetSynapses()
    {
        int count = GetDynamicSynapseCount();
        SynapseData[] result = new SynapseData[count];
        synapsePool.View.SubView(0, count).CopyToCPU(result);
        return result;
    }

    public void SetSynapses(SynapseData[] synapses)
    {
        if (synapses.Length > synapsePool.Length)
            throw new Exception(
                $"Snapshot enthält {synapses.Length} Synapsen, " +
                $"aber Poolgröße ist nur {synapsePool.Length}");

        // Synapsen in den Pool kopieren
        synapsePool.View.SubView(0, synapses.Length).CopyFromCPU(synapses);

        // Watermark setzen
        int[] watermark = new int[1];
        watermark[0] = synapses.Length;

        _synapseWatermarkBuffer.CopyFromCPU(watermark);
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

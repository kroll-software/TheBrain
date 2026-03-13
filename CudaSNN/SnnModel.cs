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

    public RenderingSystem Renderer { get; private set; } 

    public float LearningRate { get; set; } = 0.005f;

    private readonly KernelRegistry m_KernelRegistry;

    public SnnModel(string name)
    {
        Name = name;
        Gpu = new GpuEngine(); // GPU-Kontext initialisieren

        m_KernelRegistry = new KernelRegistry(Gpu.Accelerator);

        Neurons = new NeuronSystem();
        Synapses = new SynapseSystem();
        Renderer = new RenderingSystem();

        BaseSystem[] systems = [
            Neurons,
            Synapses,
            Renderer,
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

    public int CalculateMaxTotalNeurons(BrainConfiguration config)
    {
        return config.NumInputClasses * config.NumInputClassNeurons
            + config.NumHiddenLayers * config.NeuronsPerHiddenLayer
            + config.NumOutputClasses * config.NumOutputClassNeurons;        
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

        int gridDim = 32;
        float voxelSize = 20.0f; // Ein Neuron pro 10 Einheiten    
        Neurons.AutoConfigureMetrics(gridDim, voxelSize);        

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
    }    

    public void UploadToGpu(BrainConfiguration config)
    {                
        // *** Neuronen        

        Neurons.AllocateDeviceMemory(World, Gpu.Accelerator);        
        
        // 2. Synapsen Daten sammeln und hochladen        

        int maxTotalSynapses = CalculateMaxTotalSynapses(config);
        Synapses.AllocateDeviceMemory(World, maxTotalSynapses, Gpu.Accelerator);        

        Renderer.AllocateDeviceMemory(Neurons.NeuronCount, Gpu.Accelerator);
        
        // *** Synchronisieren
        Gpu.Accelerator.Synchronize();        
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
            int neuronsPerRow = (int)Neurons.WorldSize;

            switch (layerType)
            {
                case "Input":
                    // Platzierung oben: X verteilt sich, Y ist nah bei 0
                    posX = i % neuronsPerRow; 
                    posY = (i / neuronsPerRow) * 2.0f + 1f; // Mehrere Zeilen mit 2f Abstand
                    posZ = (float)Neurons.WorldSize / 2f;

                    axonX = posX;
                    axonY = (float)rand.NextDouble() * Neurons.WorldSize / 2f;
                    axonZ = (float)rand.NextDouble() * Neurons.WorldSize;
                    break;

                case "Output":
                    // Platzierung unten: X verteilt sich, Y ist nah bei WorldSize
                    posX = i % neuronsPerRow;
                    // Wir ziehen die Zeilen von der Unterkante nach oben ab
                    posY = Neurons.WorldSize - ((i / neuronsPerRow) * 2.0f -1f); 
                    posZ = (float)Neurons.WorldSize / 2f;

                    axonX = posX;
                    axonY = (Neurons.WorldSize / 2f) + ((float)rand.NextDouble() * Neurons.WorldSize / 2f);
                    axonZ = (float)rand.NextDouble() * Neurons.WorldSize;
                    break;

                default: // Hidden
                    posX = (float)rand.NextDouble() * Neurons.WorldSize;
                    posY = (float)rand.NextDouble() * Neurons.WorldSize;
                    posZ = (float)rand.NextDouble() * Neurons.WorldSize;
                    break;
            }

            if (axonX == 0 && axonY == 0 && axonZ == 0)
            {
                // Axon-Platzierung mit Mindestabstand            
                //float minDistance = Neurons.voxelSize * 1.1f; // Mindestens über die Zellgrenze hinaus
                float minDistance = Neurons.voxelSize * 3f; // Mindestens 3 Zellen Abstand
                float minDistanceSq = minDistance * minDistance;
                
                int attempts = 0;
                do
                {
                    axonX = (float)rand.NextDouble() * Neurons.WorldSize;
                    axonY = (float)rand.NextDouble() * Neurons.WorldSize;
                    axonZ = (float)rand.NextDouble() * Neurons.WorldSize;
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
            new Index1D(Neurons.NeuronCount), 
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
            new Index1D(Neurons.NeuronCount), 
            Neurons.DeviceBuffer.View, 
            Synapses.DeviceBuffer.View
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
            Neurons.sortedNeuronIDs.View,
            Neurons.gridLookup.View,
            Synapses.WatermarkBuffer.View,
            Synapses.DeviceBuffer.View,
            Neurons.gridDim,
            Neurons.voxelSize,
            Neurons.MaxReachSq,
            LearningRate
        );
    }

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
        int activeCount = Synapses.WatermarkBuffer.GetAsArray1D()[0];
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

    public NeuronRenderState[] GetNeuronPotentialsForDrawing()
    {        
        var _extractKernel = m_KernelRegistry.GetKernel<Action<Index1D, ArrayView1D<NeuronState, Stride1D.Dense>, ArrayView1D<NeuronRenderState, Stride1D.Dense>>>("ExtractRenderData");

        // 1. GPU: Daten extrahieren (dauert auf der GPU Millisekunden)
        _extractKernel(new Index1D(Neurons.NeuronCount), Neurons.DeviceBuffer.View, Renderer.DeviceBuffer.View);
        
        Renderer.DeviceBuffer.View.CopyToCPU(Renderer.RenderBuffer);
        
        return Renderer.RenderBuffer;
    }

    public void FireNeuron(int neuronIndex, float potential = 1f, float energy = 3f)
    {        
        var fireKernel = m_KernelRegistry.GetKernel<Action<Index1D, ArrayView1D<NeuronState, Stride1D.Dense>, int, float, float>>("FireNeuron");        
        fireKernel(new Index1D(Neurons.NeuronCount), Neurons.DeviceBuffer.View, neuronIndex, potential, energy);
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

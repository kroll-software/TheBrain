using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using KS.Foundation;
using KS.Foundation.ECS;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace TheBrain.CudaSNN;

public static class SnnKernels {
    // Dieser Code wird von ILGPU zu CUDA transpilliert    

    public static void UpdateNeuronStep(
        Index1D index,
        ArrayView1D<NeuronState, Stride1D.Dense> neurons,
        float energyRecoveryRate,
        int fireCycleDuration,
        int globalSeed)
    {
        var n = neurons[index];

        // --------------------------------------------------
        // Energy Recovery
        // --------------------------------------------------

        if (n.Type == 2)
            n.Energy = 1f;
        else
            n.Energy = Math.Min(1.0f, n.Energy + energyRecoveryRate);

        if (n.NewSynapseCounter > 0)
            n.NewSynapseCounter--;

        // --------------------------------------------------
        // FIRE CYCLE
        // --------------------------------------------------

        if (n.State > 0)
        {
            float spike;

            switch (n.FireCycle)
            {
                case 0: spike = 1f;   break;
                case 1: spike = 0.1f; break;
                case 2: spike = 0.8f; break;
                case 3: spike = 0.1f; break;
                case 4: spike = 0.6f; break;
                case 5: spike = 0.1f; break;
                case 6: spike = 0.4f; break;
                case 7: spike = 0.1f; break;
                default: spike = 0f; break;
            }

            // Vorzeichen für inhibitory
            if (n.Type == 1)
                spike = -spike;

            n.Output = spike;

            n.Energy -= 0.1f;

            n.FireCycle++;

            if (n.FireCycle >= fireCycleDuration)
            {
                n.State = 0;
                n.FireCycle = 0;
                n.Output = 0;
            }
        }

        // --------------------------------------------------
        // START FIRE CYCLE
        // --------------------------------------------------

        else if (n.State == 0 && n.Energy > 0.3f)
        {
            bool triggerNormal = n.Input >= n.Threshold;

            float randomVal = GpuRandom.GetRandom(index, globalSeed);
            bool triggerAuto = n.CanAutoFire == 1 && randomVal < 0.01f;

            if (triggerNormal || triggerAuto)
            {
                n.State = triggerNormal ? (byte)1 : (byte)2;
                n.FireCycle = 0;

                //float spike = (n.Type == 1) ? -1f : 1f;
                //n.Output = spike;

                n.Output = 0;
                n.Input = 0;

                n.ShortTermExcitement += 0.1f;
                n.LongTermExcitement += 0.001f;
            }
        }

        // --------------------------------------------------
        // Excitement decay
        // --------------------------------------------------

        n.ShortTermExcitement *= 0.99f;

        neurons[index] = n;
    }

    public static void FireNeuronKernel(
        Index1D index,
        ArrayView1D<NeuronState, Stride1D.Dense> neurons,
        int targetIndex,
        float input,
        float energy)
    {
        if (index != targetIndex)
            return;

        var n = neurons[index];

        n.Input = input;
        n.Energy = energy;

        n.State = 1;
        n.FireCycle = 100;

        n.Output = 1f;

        neurons[index] = n;
    }

    public static class GpuRandom
    {
        // Einfacher, schneller XorShift-Algorithmus
        public static float GetRandom(int index, int seed)
        {
            uint state = (uint)(index ^ seed);
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Konvertiere 0..uint.MaxValue zu 0.0f..1.0f
            return (float)(state & 0x7FFFFFFF) / (float)int.MaxValue;
        }
    }

    public static void ProcessPulses(
        Index1D index, 
        ArrayView1D<NeuronState, Stride1D.Dense> neurons,
        ArrayView1D<SynapseData, Stride1D.Dense> synapsePool)
    {
        // Wir arbeiten direkt auf dem Speicher
        ref var n = ref neurons[index];        
        //var n = neurons[index];        

        //n.Debug = 13;
        
        float accumulatedPotential = 0;
        int currentSynapseIdx = n.FirstSynapseIndex;

        // 1. Sammle Energie von allen verbundenen Vorgängern (Pull)
        // Wir nutzen wieder den Loop-Guard für GPU-Stabilität
        for (int i = 0; i < n.MaxSynapseLimit; i++) 
        {
            if (currentSynapseIdx == -1 || currentSynapseIdx >= synapsePool.Length)
                break;

            var synapse = synapsePool[currentSynapseIdx];
            
            // Wir holen uns den aktuellen Ladungszustand des Senders (Vorgängers)
            // Nur wenn der Sender ein Potential > 0 hat, fließt etwas
            float sourcePotential = neurons[synapse.SourceNeuronIdx].Output;
            
            if (sourcePotential > 0)
            {                                
                // Die Energieübertragung hängt vom Gewicht der Synapse ab
                accumulatedPotential += sourcePotential * synapse.Weight;
                //accumulatedPotential += sourcePotential;
            }
            
            currentSynapseIdx = synapse.NextIndex;
        }

        // 2. Integration: Das gesammelte Potential zum eigenen hinzufügen
        n.Input += accumulatedPotential;        
    }

    public static void ProcessPulses_alt(
    Index1D index, 
    ArrayView1D<NeuronState, Stride1D.Dense> neurons,
    ArrayView1D<int, Stride1D.Dense> firstSynapse,
    ArrayView1D<SynapseData, Stride1D.Dense> synapsePool,
    ArrayView1D<float, Stride1D.Dense> globalPotentials)
    {
        var n = neurons[index];        
        
        // Wenn das angesammelte Potential kritisch wird
        if (n.Input > n.Threshold)
        {
            // Wir berechnen die Puls-Stärke (unorthodox, z.B. proportional zum Überdruck)
            float pulseEnergy = (n.Input - n.Threshold) * 0.5f; 
            
            int currentSynapseIdx = firstSynapse[index];
            while (currentSynapseIdx != -1)
            {
                var synapse = synapsePool[currentSynapseIdx];
                
                // Der elektrische Impuls wandert weiter
                // Nutze AtomicAdd, da viele Pulse gleichzeitig ankommen können
                Atomic.Add(ref globalPotentials[synapse.TargetEntityID], pulseEnergy * synapse.Weight);
                
                currentSynapseIdx = synapse.NextIndex;
            }
            
            // Energieverbrauch des Neurons nach dem Feuern
            n.Input *= 0.1f; 
        }
        
        neurons[index] = n;
    }

    public static void InitFirstSynapseBuffer(Index1D index, ArrayView1D<int, Stride1D.Dense> buffer)
    {
        buffer[index] = -1;
    }

    public static unsafe void UpdateCandidateList(
        ref NeuronState n, 
        int candidateIndex, 
        int newScore)
    {
        // 1. Prüfen, ob wir überhaupt besser als das Schlusslicht sind
        if (newScore <= n.CandidateScores[7]) return;

        // 2. Prüfen, ob der Kandidat schon drin ist (Update statt Insert)
        for (int i = 0; i < 8; i++)
        {
            if (n.CandidateIndices[i] == candidateIndex)
            {
                n.CandidateScores[i] = newScore;
                // Nach oben "hochblubbern" (Insertion-Sort Logik)
                while (i > 0 && n.CandidateScores[i] > n.CandidateScores[i - 1])
                {
                    Swap(ref n, i, i - 1);
                    i--;
                }
                return;
            }
        }

        // 3. Wenn nicht vorhanden: Neuen Kandidaten am Ende einfügen
        n.CandidateIndices[7] = candidateIndex;
        n.CandidateScores[7] = newScore;

        // 4. In die richtige Position einsortieren
        int j = 6;
        while (j >= 0 && n.CandidateScores[j + 1] > n.CandidateScores[j])
        {
            Swap(ref n, j, j + 1);
            j--;
        }
    }

    public static unsafe void RemoveFromCandidateList(ref NeuronState n, int candidateIndex)
    {
        int foundIndex = -1;

        // 1. Suche den Kandidaten in der Liste
        for (int i = 0; i < 8; i++)
        {
            if (n.CandidateIndices[i] == candidateIndex)
            {
                foundIndex = i;
                break;
            }
        }

        // 2. Wenn gefunden, löschen und Rest nachrücken lassen
        if (foundIndex != -1)
        {
            // Verschiebe alle nachfolgenden Kandidaten um einen Platz nach oben
            for (int i = foundIndex; i < 7; i++)
            {
                n.CandidateIndices[i] = n.CandidateIndices[i + 1];
                n.CandidateScores[i] = n.CandidateScores[i + 1];
            }

            // Den letzten Platz (Nr. 7) sauber "nullen"
            n.CandidateIndices[7] = -1;
            n.CandidateScores[7] = 0;
        } else
        {
            n.Debug = 12;
        }
    }

    // Hilfsfunktion zum Tauschen der Arrays innerhalb des Structs
    private static unsafe void Swap(ref NeuronState n, int a, int b)
    {
        int tempIdx = n.CandidateIndices[a];
        int tempScore = n.CandidateScores[a];
        
        n.CandidateIndices[a] = n.CandidateIndices[b];
        n.CandidateScores[a] = n.CandidateScores[b];
        
        n.CandidateIndices[b] = tempIdx;
        n.CandidateScores[b] = tempScore;
    }    

    public static void DebugSetKernel(Index1D index, 
        ArrayView1D<NeuronState, 
        Stride1D.Dense> neurons)
    {
        var n = neurons[index];
        n.Debug = 99; // Setze einen eindeutigen Wert
        neurons[index] = n;
    }


    public static unsafe void HebbianSpatialKernelFast(
        Index1D index,
        ArrayView1D<NeuronState, Stride1D.Dense> neurons,
        ArrayView1D<int, Stride1D.Dense> sortedNeuronIDs,
        ArrayView1D<int, Stride1D.Dense> gridLookup,
        ArrayView1D<int, Stride1D.Dense> watermarkBuffer,
        ArrayView1D<SynapseData, Stride1D.Dense> synapsePool,
        int gridDim,
        float voxelSize,
        float maxReachSq,
        float learningRate)
    {
        //int neuronIndex = index.GridIdx;
        int neuronIndex = index;
        if (neuronIndex >= neurons.Length)
            return;

        ref var receiver = ref neurons[neuronIndex];

        int cx = (int)(receiver.PosX / voxelSize);
        int cy = (int)(receiver.PosY / voxelSize);
        int cz = (int)(receiver.PosZ / voxelSize);

        cx = XMath.Clamp(cx, 0, gridDim - 1);
        cy = XMath.Clamp(cy, 0, gridDim - 1);
        cz = XMath.Clamp(cz, 0, gridDim - 1);

        // Shared memory Tile
        var tile = ILGPU.SharedMemory.Allocate<NeuronState>(Group.Dimension.X);

        // Nachbarschaftszellen
        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++)
        {
            int nx = cx + dx;
            int ny = cy + dy;
            int nz = cz + dz;

            if (nx < 0 || ny < 0 || nz < 0 ||
                nx >= gridDim || ny >= gridDim || nz >= gridDim)
                continue;

            int cell = nx + gridDim * ny + gridDim * gridDim * nz;

            int start = gridLookup[cell];
            int end   = gridLookup[cell + 1];

            // Tile Iteration
            for (int tileStart = start; tileStart < end; tileStart += Group.Dimension.X)
            {
                int tileIndex = tileStart + Group.IdxX;

                if (tileIndex < end)
                    tile[Group.IdxX] = neurons[sortedNeuronIDs[tileIndex]];

                Group.Barrier();

                int tileSize = XMath.Min(Group.Dimension.X, end - tileStart);

                for (int i = 0; i < tileSize; i++)
                {
                    var source = tile[i];

                    if (receiver.State <= 0 || source.State <= 0)
                        continue;

                    float dxDist = source.AxonX - receiver.PosX;
                    float dyDist = source.AxonY - receiver.PosY;
                    float dzDist = source.AxonZ - receiver.PosZ;

                    float distSq =
                        dxDist * dxDist +
                        dyDist * dyDist +
                        dzDist * dzDist;

                    if (distSq >= maxReachSq)
                        continue;

                    bool found =
                        ReinforceExistingSynapse(
                            ref receiver,
                            source.ID,
                            synapsePool,
                            learningRate);

                    if (!found)
                    {
                        UpdateCandidateList(ref receiver, source.ID, 1);

                        if (receiver.CandidateScores[0] > 0.01f)
                        {
                            AttemptSynapseGrowth(
                                ref receiver,
                                source.ID,
                                neurons,
                                watermarkBuffer,
                                synapsePool);
                        }
                    }
                }

                Group.Barrier();
            }
        }
    }    

    public static unsafe void HebbianSpatialKernel(
        Index1D index,
        ArrayView1D<NeuronState, Stride1D.Dense> neurons,
        ArrayView1D<int, Stride1D.Dense> sortedNeuronIDs,
        ArrayView1D<int, Stride1D.Dense> gridLookup,
        ArrayView1D<int, Stride1D.Dense> watermarkBuffer, // Muss hier rein
        ArrayView1D<SynapseData, Stride1D.Dense> synapsePool, // Muss hier rein
        int gridDim,
        float voxelSize,
        float maxReachSq,
        float learningRate)
    {        
        ref var receiver = ref neurons[index];
        
        //neurons[index].Debug = gridLookup[0];

        // 1. Bestimme mein Voxel (wo ist mein Soma?)
        int cx = (int)(receiver.PosX / voxelSize);
        int cy = (int)(receiver.PosY / voxelSize);
        int cz = (int)(receiver.PosZ / voxelSize);        

        // 2. Suche in der 3x3x3 Nachbarschaft nach Axon-Enden
        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++)
        {
            int nx = cx + dx;
            int ny = cy + dy;
            int nz = cz + dz;

            // Boundary Check: Nur innerhalb des Grids suchen
            if (nx >= 0 && nx < gridDim && ny >= 0 && ny < gridDim && nz >= 0 && nz < gridDim)            
            {                
                int neighborCell = nx + (gridDim * ny) + (gridDim * gridDim * nz);
                if (neighborCell >= 0 && neighborCell < gridLookup.Length - 1)            
                {                           
                    // Hole Bereich aus dem sortierten Array
                    int start = gridLookup[neighborCell];
                    int end = gridLookup[neighborCell + 1];                    
                    
                    for (int k = start; k < end; k++)
                    {
                        int sourceIdx = sortedNeuronIDs[k];
                        var source = neurons[sourceIdx];   

                        if (sourceIdx == index)
                            continue;

                        // Hebb'sche Bedingung: Beide müssen feuern (State == 1)
                        if (receiver.State > 0 && source.State > 0)
                        {                            
                            // Distanz Axon(Sender) zu Soma(Receiver)
                            float dxDist = source.AxonX - receiver.PosX;
                            float dyDist = source.AxonY - receiver.PosY;
                            float dzDist = source.AxonZ - receiver.PosZ;
                            float distSq = (dxDist * dxDist) + (dyDist * dyDist) + (dzDist * dzDist);
                            
                            // Lernen, wenn innerhalb der Axon-Reichweite
                            if (distSq < maxReachSq)
                            {                                
                                bool synapticLinkFound = ReinforceExistingSynapse(ref receiver, sourceIdx, synapsePool, learningRate);
                                
                                if (!synapticLinkFound) 
                                {                                    

                                    // Hier rufen wir deine Logik auf
                                    UpdateCandidateList(ref receiver, sourceIdx, 1);                                    

                                    // 2. Prüfung: Ist der Score für diesen Kandidaten hoch genug?
                                    // Hinweis: Du musst hier ggf. den Score des speziellen Kandidaten 
                                    // aus receiver.CandidateScores[k] abrufen
                                    if (receiver.CandidateScores[0] > 0.01f)
                                    {
                                        //neurons[index].Debug = 12;                                        

                                        // 3. Wachstum versuchen
                                        // Wir übergeben das Neuron und den aktuellen sourceIdx
                                        AttemptSynapseGrowth(ref receiver, sourceIdx, neurons, watermarkBuffer, synapsePool);
                                    }                                    
                                }

                                neurons[index] = receiver;
                            }
                        }
                    }
                }
            }
        }
    }    
    
    private static void AttemptSynapseGrowth_old(
        ref NeuronState receiver, 
        int sourceIdx, 
        ArrayView1D<int, Stride1D.Dense> watermarkBuffer, // Dein globaler Pool-Zähler
        ArrayView1D<SynapseData, Stride1D.Dense> synapsePool)
    {
        // 1. Hole einen neuen Slot aus dem Pool
        int newIdx = Atomic.Add(ref watermarkBuffer[0], 1);
        
        if (newIdx < synapsePool.Length)
        {
            // 2. Erstelle den neuen Knoten
            // Der neue Knoten zeigt auf das bisherige Head des Neurons
            int oldHead = Atomic.Exchange(ref receiver.FirstSynapseIndex, newIdx);
            
            synapsePool[newIdx] = new SynapseData { 
                TargetEntityID = sourceIdx, // Das Neuron, mit dem wir wachsen
                Weight = 0.1f,
                NextIndex = oldHead // Linkt auf das alte Head
            };
        }
    }

    private static void AttemptSynapseGrowth(
        ref NeuronState receiver, 
        int sourceIdx, 
        ArrayView1D<NeuronState, Stride1D.Dense> allNeurons, // Neu hinzugefügt
        ArrayView1D<int, Stride1D.Dense> watermarkBuffer, // Dein globaler Pool-Zähler
        ArrayView1D<SynapseData, Stride1D.Dense> synapsePool)
    {   
        if (receiver.ID == sourceIdx)
            return;

        if (receiver.NewSynapseCounter > 0)
            return;

        // PRÜFUNG: Gibt es schon eine Verbindung in die Gegenrichtung?
        if (HasExistingReverseConnection(receiver.ID, sourceIdx, allNeurons, synapsePool))
        {
            // Falls ja: Abbruch und aus der Kandidatenliste werfen, 
            // damit wir es nicht sofort wieder versuchen.
            RemoveFromCandidateList(ref receiver, sourceIdx);
            return;
        }

        // In AttemptSynapseGrowth:
        int poolIdx = Atomic.Add(ref watermarkBuffer[0], 1);
        if (poolIdx >= synapsePool.Length)
        {
            Atomic.Add(ref watermarkBuffer[0], -1);
            return;
        }

        if (poolIdx < synapsePool.Length)
        {
            // 1. Die neue Synapse vorbereiten
            synapsePool[poolIdx] = new SynapseData {
                SourceNeuronIdx = sourceIdx,
                TargetEntityID = receiver.ID,
                Weight = 0.1f,
                // 2. WICHTIG: Die neue Synapse zeigt auf den bisherigen Kopf der Liste
                NextIndex = Atomic.Exchange(ref receiver.FirstSynapseIndex, poolIdx)
            };
            receiver.CurrentSynapseCount++;
            receiver.NewSynapseCounter = 100;
            receiver.Energy = 0;
            RemoveFromCandidateList(ref receiver, sourceIdx);
        }        
    }

    private static bool HasExistingReverseConnection(
    int myID,
    int potentialSourceIdx,
    ArrayView1D<NeuronState, Stride1D.Dense> allNeurons,
    ArrayView1D<SynapseData, Stride1D.Dense> synapsePool)
    {
        // Wir prüfen Synapsen die von meinem Neuron ausgehen
        int currentSynIdx = allNeurons[myID].FirstSynapseIndex;

        int safetyCounter = 0;

        while (currentSynIdx != -1 && safetyCounter < 1000)
        {
            if (currentSynIdx >= synapsePool.Length)
                break;

            ref SynapseData syn = ref synapsePool[currentSynIdx];

            if (syn.SourceNeuronIdx == myID &&
                syn.TargetEntityID == potentialSourceIdx)
            {
                return true; // myID -> source existiert bereits
            }

            currentSynIdx = syn.NextIndex;
            safetyCounter++;
        }

        return false;
    }
    
    private static bool ReinforceExistingSynapse(
        ref NeuronState receiver, 
        int sourceIdx, 
        ArrayView1D<SynapseData, Stride1D.Dense> synapsePool,
        float learningRate)
    {
        int currentIdx = receiver.FirstSynapseIndex;
                
        //for (int i = 0; i < 64; i++) 
        while (true)    // all synapses
        {
            if (currentIdx == -1 || currentIdx >= synapsePool.Length)
                break;

            ref SynapseData syn = ref synapsePool[currentIdx];
                        
            if (syn.SourceNeuronIdx == sourceIdx)
            {                
                syn.Weight += learningRate * syn.Weight * (1f - syn.Weight);
                return true;
            }
            
            currentIdx = syn.NextIndex;
        }
        return false;
    }

    // Innerhalb eines Kernels, der Distanz prüft
    public static float GetDistance(float x1, float y1, float z1, float x2, float y2, float z2)
    {
        float dx = x1 - x2;
        float dy = y1 - y2;
        float dz = z1 - z2;
        // sqrt ist auf der GPU teuer, falls möglich: arbeite mit quadratischer Distanz
        return (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
    }    

    // Der Kernel für die GPU
    public static void CalculateWeightStatsKernel(
        Index1D index,
        ArrayView1D<SynapseData, Stride1D.Dense> synapsePool,
        int activeSynapseCount, // Dies ist der Wert aus deinem watermarkBuffer[0]
        ArrayView1D<WeightSumResult, Stride1D.Dense> resultBuffer)
    {
        // Wir summieren nur bis zum watermark-Stand
        if (index >= activeSynapseCount) return;

        var synapse = synapsePool[index];
        
        // Wir nutzen Atomics, um die Summe im resultBuffer[0] zu bilden
        // Hinweis: Atomic.Add für float benötigt in ILGPU oft eine spezielle Extension oder ein Int-Bit-Mapping
        // Einfacher für den Anfang: Wir schreiben das Ergebnis pro Warp/Block (hier vereinfacht direkt)
        Atomic.Add(ref resultBuffer[0].TotalWeight, synapse.Weight);
        Atomic.Add(ref resultBuffer[0].ActiveCount, 1);
    }    

    // Kernel: Zählt wie viele Neuronen gerade feuern
    public static void CountActiveNeurons(
        Index1D index, 
        ArrayView1D<NeuronState, Stride1D.Dense> neurons, 
        ArrayView1D<int, Stride1D.Dense> activeCount)
    {
        if (neurons[index].State > 0)
        {
            Atomic.Add(ref activeCount[0], 1);
        }
    }

    public static void ExtractRenderDataKernel(
    Index1D index, 
    ArrayView1D<NeuronState, Stride1D.Dense> source, 
    ArrayView1D<NeuronRenderState, Stride1D.Dense> target)
    {
        var s = source[index];
        target[index] = new NeuronRenderState {            
            Type = s.Type,
            State = s.State,
            Output = s.Output,
            PosX = s.PosX,
            PosY = s.PosY,
            PosZ = s.PosZ,
            Debug = s.Debug
        };
    }

    public static void CountActiveSynapsesKernel(
        Index1D index,
        ArrayView1D<SynapseData, Stride1D.Dense> synapses,
        ArrayView1D<int, Stride1D.Dense> counter)
        {
            // Zugriff über das Feld 'Weight' in deinem SynapseData-Struct
            if (synapses[index].Weight != 0.0f)
            {
                Atomic.Add(ref counter[0], 1);
            }
        }
}

public static class NeuronKernels
{
    public static void ApplyDecay(
        Index1D index, 
        ArrayView1D<NeuronState, Stride1D.Dense> states, 
        float decayFactor)
    {
        // Zugriff auf das Neuron an dieser Position
        NeuronState n = states[index];
        
        // Simpler Decay
        n.Input *= decayFactor;
        
        // Zurückschreiben
        states[index] = n;
    }    
}

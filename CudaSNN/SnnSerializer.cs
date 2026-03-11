using System;
//using System.Text.Json; // Für JsonSerializer
using System.Runtime.InteropServices; // Für MemoryMarshal
using System.IO; // Für FileStream und BinaryWriter
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using KS.Foundation;
using KS.Foundation.ECS;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Diagnostics;

namespace TheBrain.CudaSNN;

public class NeuronStateConverter : JsonConverter<NeuronState>
{
    public override void WriteJson(JsonWriter writer, NeuronState value, JsonSerializer serializer)
    {
        writer.WriteStartObject();

        // Einfache Felder
        writer.WritePropertyName("ID"); writer.WriteValue(value.ID);
        writer.WritePropertyName("Input"); writer.WriteValue(value.Input);
        writer.WritePropertyName("Output"); writer.WriteValue(value.Output);
        writer.WritePropertyName("Threshold"); writer.WriteValue(value.Threshold);
        writer.WritePropertyName("Energy"); writer.WriteValue(value.Energy);
        writer.WritePropertyName("ShortTermExcitement"); writer.WriteValue(value.ShortTermExcitement);
        writer.WritePropertyName("LongTermExcitement"); writer.WriteValue(value.LongTermExcitement);
        writer.WritePropertyName("ConnectionRadius"); writer.WriteValue(value.ConnectionRadius);
        writer.WritePropertyName("FirstSynapseIndex"); writer.WriteValue(value.FirstSynapseIndex);
        writer.WritePropertyName("CurrentSynapseCount"); writer.WriteValue(value.CurrentSynapseCount);
        writer.WritePropertyName("MaxSynapseLimit"); writer.WriteValue(value.MaxSynapseLimit);
        writer.WritePropertyName("NewSynapseCounter"); writer.WriteValue(value.NewSynapseCounter);
        writer.WritePropertyName("PosX"); writer.WriteValue(value.PosX);
        writer.WritePropertyName("PosY"); writer.WriteValue(value.PosY);
        writer.WritePropertyName("PosZ"); writer.WriteValue(value.PosZ);
        writer.WritePropertyName("AxonX"); writer.WriteValue(value.AxonX);
        writer.WritePropertyName("AxonY"); writer.WriteValue(value.AxonY);
        writer.WritePropertyName("AxonZ"); writer.WriteValue(value.AxonZ);
        writer.WritePropertyName("FireCycleRemaining"); writer.WriteValue(value.FireCycle);
        writer.WritePropertyName("Type"); writer.WriteValue(value.Type);
        writer.WritePropertyName("State"); writer.WriteValue(value.State);
        writer.WritePropertyName("IsAutoFireActive"); writer.WriteValue(value.CanAutoFire);
        writer.WritePropertyName("Debug"); writer.WriteValue(value.Debug);

        // Fixed Arrays als echte JSON-Arrays
        unsafe
        {
            writer.WritePropertyName("CandidateIndices");
            writer.WriteStartArray();
            for (int i = 0; i < 8; i++) writer.WriteValue(value.CandidateIndices[i]);
            writer.WriteEndArray();

            writer.WritePropertyName("CandidateScores");
            writer.WriteStartArray();
            for (int i = 0; i < 8; i++) writer.WriteValue(value.CandidateScores[i]);
            writer.WriteEndArray();
        }

        writer.WriteEndObject();
    }

    public override NeuronState ReadJson(JsonReader reader, Type objectType, NeuronState existingValue, bool hasExistingValue, JsonSerializer serializer)
    {
        JObject jo = JObject.Load(reader);
        NeuronState state = new NeuronState();

        // Einfache Felder mit Null-Prüfung/Default-Wert-Logik
        state.ID = jo["ID"]?.Value<int>() ?? 0;
        state.Input = jo["Input"]?.Value<float>() ?? 0f;
        state.Output = jo["Output"]?.Value<float>() ?? 0f;
        state.Threshold = jo["Threshold"]?.Value<float>() ?? 0f;
        state.Energy = jo["Energy"]?.Value<float>() ?? 0f;
        state.ShortTermExcitement = jo["ShortTermExcitement"]?.Value<float>() ?? 0f;
        state.LongTermExcitement = jo["LongTermExcitement"]?.Value<float>() ?? 0f;
        state.ConnectionRadius = jo["ConnectionRadius"]?.Value<float>() ?? 0f;
        state.FirstSynapseIndex = jo["FirstSynapseIndex"]?.Value<int>() ?? -1;
        state.CurrentSynapseCount = jo["CurrentSynapseCount"]?.Value<int>() ?? 0;
        state.MaxSynapseLimit = jo["MaxSynapseLimit"]?.Value<int>() ?? 0;
        state.NewSynapseCounter = jo["NewSynapseCounter"]?.Value<int>() ?? 0;
        state.PosX = jo["PosX"]?.Value<float>() ?? 0f;
        state.PosY = jo["PosY"]?.Value<float>() ?? 0f;
        state.PosZ = jo["PosZ"]?.Value<float>() ?? 0f;
        state.AxonX = jo["AxonX"]?.Value<float>() ?? 0f;
        state.AxonY = jo["AxonY"]?.Value<float>() ?? 0f;
        state.AxonZ = jo["AxonZ"]?.Value<float>() ?? 0f;
        state.FireCycle = jo["FireCycleRemaining"]?.Value<int>() ?? 0;
        state.Type = jo["Type"]?.Value<byte>() ?? 0;
        state.State = jo["State"]?.Value<byte>() ?? 0;
        state.CanAutoFire = jo["IsAutoFireActive"]?.Value<byte>() ?? 0;
        state.Debug = jo["Debug"]?.Value<int>() ?? 0;

        // Fixed Arrays sicher befüllen
        unsafe
        {
            var indices = jo["CandidateIndices"] as JArray;
            if (indices != null)
                for (int i = 0; i < Math.Min(indices.Count, 8); i++) state.CandidateIndices[i] = indices[i].Value<int>();

            var scores = jo["CandidateScores"] as JArray;
            if (scores != null)
                for (int i = 0; i < Math.Min(scores.Count, 8); i++) state.CandidateScores[i] = scores[i].Value<int>();
        }

        return state;
    }
}

public class SynapseDataConverter : JsonConverter<SynapseData>
{
    public override void WriteJson(JsonWriter writer, SynapseData value, JsonSerializer serializer)
    {
        writer.WriteStartObject();
        writer.WritePropertyName("SourceNeuronIdx"); writer.WriteValue(value.SourceNeuronIdx);
        writer.WritePropertyName("TargetEntityID"); writer.WriteValue(value.TargetEntityID);
        writer.WritePropertyName("Weight"); writer.WriteValue(value.Weight);
        writer.WritePropertyName("NextIndex"); writer.WriteValue(value.NextIndex);
        writer.WriteEndObject();
    }

    public override SynapseData ReadJson(JsonReader reader, Type objectType, SynapseData existingValue, bool hasExistingValue, JsonSerializer serializer)
    {
        JObject jo = JObject.Load(reader);
        
        return new SynapseData
        {
            SourceNeuronIdx = jo["SourceNeuronIdx"]?.Value<int>() ?? -1,
            TargetEntityID = jo["TargetEntityID"]?.Value<int>() ?? -1,
            Weight = jo["Weight"]?.Value<float>() ?? 0.0f,
            NextIndex = jo["NextIndex"]?.Value<int>() ?? -1
        };
    }
}


public class SnnSerializer
{

    public void Save(string filePath, SnnModel model)
    {
        SynapseData[] synapses = model.GetSynapses();

        // 1. Snapshot-Objekt füllen
        var snapshot = new ModelSnapshot {
            Iteration = model.Iteration,
            Neurons = model.Neurons.DeviceBuffer.GetAsArray1D(),
            Synapses = synapses
        };

        // 2. Settings mit Convertern konfigurieren
        var settings = new JsonSerializerSettings
        {
            Formatting = Formatting.None,
            Converters = { 
                new NeuronStateConverter(),
                new SynapseDataConverter()
            }
        };

        // 3. Serialisieren mit dem DTO-Typ
        using (StreamWriter sw = new StreamWriter(filePath))
        using (JsonTextWriter writer = new JsonTextWriter(sw))
        {
            JsonSerializer serializer = JsonSerializer.Create(settings);
            serializer.Serialize(writer, snapshot);
        }

        // Some statistics ...
        float maxWeight = synapses.Max(s => s.Weight);
        float avgWeight = synapses.Average(s => s.Weight);

        Debug.WriteLine($"Synapses Max Weight: {maxWeight}, Avg. Weight: {avgWeight}");

    }

    public class ModelSnapshot
    {
        public NeuronState[] Neurons { get; set; }
        public SynapseData[] Synapses { get; set; }
        public int Iteration { get; set; }
    }

    public void Load(string filePath, SnnModel model)
    {
        var settings = new JsonSerializerSettings {
            Converters = { new NeuronStateConverter(), new SynapseDataConverter() }
        };

        using (StreamReader sr = new StreamReader(filePath))
        using (JsonTextReader reader = new JsonTextReader(sr))
        {
            var serializer = JsonSerializer.Create(settings);
            var snapshot = serializer.Deserialize<ModelSnapshot>(reader);

            if (snapshot == null)
                throw new Exception("Snapshot-Datei konnte nicht geladen werden.");

            if (snapshot.Neurons.Length != model.NeuronCount)
                throw new Exception(
                    $"Konfigurations-Mismatch: Snapshot enthält {snapshot.Neurons.Length} " +
                    $"Neuronen, aber GPU erwartet {model.NeuronCount}.");

            // Iteration wiederherstellen
            model.Iteration = snapshot.Iteration;

            // Neuronen laden
            model.Neurons.DeviceBuffer.CopyFromCPU(snapshot.Neurons);

            // Synapsen laden (über neue Model-Funktion)
            model.SetSynapses(snapshot.Synapses);
        }
    }

    /***
    public void SaveBinary(string folderPath, SnnModel model)
    {
        // 1. Metadaten als JSON (Konfiguration, Anzahl der Neuronen etc.)
        var meta = new {
            model.NeuronCount,
            model.SynapseCount,
            Timestamp = DateTime.Now
        };
        File.WriteAllText(Path.Combine(folderPath, "meta.json"), JsonSerializer.Serialize(meta));

        // 2. Binäre Dumps der GPU-Buffer
        // Wir kopieren die Daten kurz in ein CPU-Array und schreiben sie
        SaveBuffer(Path.Combine(folderPath, "neurons.bin"), model.Neurons.DeviceBuffer);
        SaveBuffer(Path.Combine(folderPath, "synapses.bin"), model.Synapses.DeviceBuffer);
    }

    private void SaveBuffer<T>(string path, MemoryBuffer1D<T, Stride1D.Dense> buffer) where T : unmanaged
    {
        // 1. Snapshot auf CPU ziehen
        T[] data = buffer.GetAsArray1D(); 
        
        // 2. Binär in Datei schreiben
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        ReadOnlySpan<T> span = data.AsSpan();
        ReadOnlySpan<byte> byteSpan = MemoryMarshal.AsBytes(span);
        fs.Write(byteSpan);
    }    
    ***/
}


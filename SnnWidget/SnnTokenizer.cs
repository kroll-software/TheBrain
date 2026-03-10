using System;
//using Microsoft.ML.Tokenizers;
using Tiktoken;

namespace TheBrain;

public class SnnTokenizer
{
    private readonly Tiktoken.Encoder tokenizer;
    private readonly int _maxNeuronId;

    public int VocabSize {get; private set;}

    public SnnTokenizer(int maxNeuronId)
    {        
        var encoding = new Tiktoken.Encodings.Cl100KBase();
        tokenizer = new Tiktoken.Encoder(encoding);

        _maxNeuronId = maxNeuronId;

        //VocabSize = Math.Min(maxNeuronId, _tokenizer.);
    }

    public int[] EncodeAndMap(string text)
    {        
        var ids = tokenizer.Encode(text);
        return ids.Select(id => (int)(id % _maxNeuronId)).ToArray();
    }
}
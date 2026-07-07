using System;
using Microsoft.ML.Tokenizers;

//using Microsoft.ML.Tokenizers;
using Tiktoken;

namespace TheBrain;

public class SnnTokenizer
{
    //private readonly Tiktoken.Encoder tokenizer;
    private readonly BpeTokenizer tokenizer;    

    public int VocabSize {get; private set;}

    public SnnTokenizer()
    {
        string vocab_path = "./tokenizer/vocab.json";
        string merge_path = "./tokenizer/merges.txt";
    
        if (!File.Exists(vocab_path) || !File.Exists(merge_path))
        {
            throw new FileNotFoundException("Tokenizer files missing! Please download vocab.json and merges.txt from Hugging Face and place them in the /tokenizer folder.");
        }

        tokenizer = BpeTokenizer.Create(vocab_path, merge_path);
        VocabSize = tokenizer.Vocabulary.Count;        
    }

    public int[] EncodeAndMap(string text)
    {        
        return tokenizer.EncodeToIds(text).ToArray();
    }
}
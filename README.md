# 🧠 The Brain (Experimental)

Welcome to **The Brain**! This is an experimental, high-performance **Spiking Neural Network (SNN)** engine designed to simulate biologically inspired learning processes directly on the GPU.

> ⚠️ **Status: Work in Progress**
> This project is in an early experimental phase. The code is built for research and rapid prototyping of neural architectures.

---

## 🚀 Key Features

* **GPU-First Architecture:** Utilizes **OpenTK 4.x** and **GLFW** to compute massive neural networks in parallel using Compute Shaders.
* **ECS-Inspired Design:** Efficient management of `NeuronState` and `SynapseData` structures to maximize performance for hundreds of thousands of neurons.
* **Custom Serialization:** Robust, high-speed snapshotting via **Newtonsoft.Json**, featuring custom converters to handle C# `unsafe fixed` buffers efficiently.
* **Cross-Platform:** Developed for **Ubuntu** and **Windows** using VS-Code.
* **Self-Wiring Capabilities:** Includes built-in mechanisms for structural plasticity and synapse candidate ranking.


---

## 💻 System Requirements

To ensure smooth simulation and enough memory for the neural pool, **The Brain** requires:

* **GPU:** A CUDA-compatible graphics card (NVIDIA).
* **VRAM:** At least **8 GB** of dedicated video memory.
* **OS:** Windows 10/11 or Ubuntu 20.04+ (tested with VS-Code).



> 🚀 **Note:** While the engine is highly optimized, the large input/output layers and the 400k-neuron hidden pool thrive on fast VRAM and high CUDA core counts.

---

## 🛠 Tech Stack

* **Language:** C#
* **Graphics/Compute:** OpenTK (OpenGL/Compute Shaders)
* **Serialization:** Newtonsoft.Json (Custom Converters for `unsafe fixed` memory layouts)
* **IDE:** VS-Code

---

## 📦 Snapshot System
**The Brain** utilizes an optimized snapshot system to persist the neural state (including `Iteration`, `Neurons`, and `Synapses`). By implementing custom JSON converters, complex GPU memory structures are mapped into clean, human-readable JSON formats without losing data integrity.



---

## 📋 Roadmap (WIP)

- [ ] Refinement of Hebbian learning kernels.
- [ ] Expansion of the custom widget hierarchy for better visualization.
- [ ] Further optimization of the `WM_LBUTTONDOWN` activation fix.
- [ ] API documentation for external training dataset ingestion.

---

## 🛠 Setup & Tokenizer

To run **The Brain**, you need the GPT-2 tokenizer files. Due to licensing and size, they are not included in this repository. Please follow these steps to set up your environment:

1.  Create a folder named `tokenizer` in the project root directory.
2.  Download the following two files from the official Hugging Face GPT-2 repository:
    * [`vocab.json`](https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json)
    * [`merges.txt`](https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt)
3.  Place both files into the `/tokenizer` folder you created.

The system will automatically detect and load these files upon the first build.



> 💡 **Pro-Tip:** If the files are missing, the engine will prompt you with a `FileNotFoundException` to ensure you have the necessary dependencies for the BPE-based tokenization.

---

## 🤝 Contribution & Feedback

As this project is experimental, I welcome constructive discussions and feedback regarding:
* GPU memory optimization.
* SNN topology research.
* Serialization efficiency for large-scale buffers.

*Please note: Interfaces and data structures are subject to frequent changes.*

---
*Built with ❤️ and a lot of C#.*
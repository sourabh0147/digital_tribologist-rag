# Digital Tribologist: Localized RAG Framework for Advanced Manufacturing

## Description
The "Digital Tribologist" is a localized, edge-capable Retrieval-Augmented Generation (RAG) framework designed to function as an expert manufacturing consultant. It parses dense, domain-specific literature in advanced manufacturing, high-entropy alloys (HEAs), and laser processing. Built for proprietary manufacturing environments, it prioritizes data privacy and operates without reliance on external commercial APIs. 

The system architecture utilizes the `microsoft/Phi-3-mini-4k-instruct` base model, aggressively optimized via 4-bit NormalFloat (NF4) quantization to operate securely within the 16GB VRAM constraint of a single Nvidia T4 GPU. The ingestion pipeline ensures the structural integrity of dense mathematical formulations and technical tables, particularly those inherent to Multi-Output Random Forest surrogate frameworks.

## Key Features
* **Absolute Data Sovereignty:** Fully localized execution ensures that no proprietary alloy compositions, laser processing parameters, or source documents leave the secure edge or private cloud instance.
* **High-Fidelity Contextual Integrity:** Utilizes `LangChain` and `FAISS` with a custom chunking heuristic (size: 800, overlap: 100) to prevent the fragmentation of multi-line mathematical formulations and empirical data tables.
* **Hardware Efficiency:** Democratizes access to AI consultation by running state-of-the-art deductive reasoning on ubiquitous, low-cost hardware accelerators (Nvidia T4) rather than expensive multi-GPU clusters.
* **Deterministic Output:** Suppresses hallucinations using strict hyperparameter boundaries (temperature of 0.1, repetition penalty of 1.1), forcing the LLM into a highly deterministic regime suitable for academic and industrial consultation.
* **Emergency Runtime Patching:** Includes engineered Python metaprogramming to resolve dependency desynchronization between Hugging Face and `bitsandbytes` backend validation in cloud notebook environments.

## Requirements
The framework requires a single GPU with at least 16GB VRAM (e.g., Nvidia T4). 

**Core Dependencies:**
* `Python >= 3.12`
* `torch`
* `transformers >= 4.44.0`
* `bitsandbytes >= 0.46.1`
* `accelerate >= 0.33.0`
* `langchain == 0.2.16`
* `faiss-cpu >= 1.8.0.post1`
* `sentence-transformers >= 3.0.1`
* `gradio >= 4.42.0`
* `numpy >= 2.0.0, < 2.1.0`

*(Note: Strict dependency version-pinning is necessary to prevent library desynchronization in ephemeral cloud environments.)*

## Usage
1. **Upload:** Launch the Gradio UI and upload a research paper (PDF) related to manufacturing or tribology.
2. **Ingest:** The system uses `sentence-transformers/all-MiniLM-L6-v2` to vectorize the text and indices it in-memory using FAISS.
3. **Query:** Ask domain-specific questions. The system retrieves the top 3 semantically relevant chunks ($k=3$) and dynamically injects them into the Phi-3 prompt template to generate an expert-level recommendation.

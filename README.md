# Hybrid Retrieval-Augmented Generation (RAG) System for Document Question Answering

This project implements a production-style **Retrieval-Augmented Generation (RAG)** system that enables users to ask questions over documents and receive accurate, context-grounded answers using a deployed **Large Language Model (LLM)**.

The system combines document retrieval and language generation to ensure responses are grounded in the source documents rather than relying solely on the LLM’s parametric knowledge.

---

## Key Features

- Hybrid retrieval combining **semantic (embedding-based) search** and **keyword-based search**
- Document-based question answering over **long-form, multi-page PDF documents**
- Context-aware answer generation using a **deployed Large Language Model (LLM)**
- End-to-end pipeline including document processing, retrieval, and answer generation
- Interactive **web-based interface** for real-time querying

---

## High-Level Architecture

1. Documents are ingested and processed into manageable chunks  
2. Relevant chunks are retrieved using a **hybrid retrieval strategy**  
3. Retrieved context is provided to an LLM for answer generation  
4. Answers are returned to the user via a web interface  

This approach improves factual accuracy and reduces hallucinations compared to standalone LLMs.

---

## Tech Stack

- **Python**
- **PyTorch**
- **Retrieval-Augmented Generation (RAG)**
- **Sentence Transformers** (for semantic embeddings)
- **Docker** (LLM deployment)
- **Gradio** (interactive web UI)
- **REST APIs**
- **CUDA / GPU acceleration** (optional)

---

## Repository Structure

```text
.
├── app.py
├── ui.py
├── rethinker_retrieval.py
├── generation.py
├── hf_generation.py
├── vllm_generation.py
├── torchvectorbase.py
├── torchchuck.py
├── conversion.py
├── history.py
├── ragconfig.yaml
├── requirements.txt
├── requirements1.txt
├── data/        # Runtime-generated data (ignored in version control)
├── out/         # Runtime outputs (ignored in version control)

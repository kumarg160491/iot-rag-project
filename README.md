IoT Telemetry RAG for Predictive Maintenance
==================================================

A production-style Retrieval Augmented Generation (RAG) system that combines IoT domain knowledge with a locally running Large Language Model to diagnose machine faults and suggest corrective actions — with zero internet dependency at runtime.

Tech Stack
=============
Layer                               Tool
LLM: Llama 3.2 via Ollama
Embeddings: nomic-embed-text via Ollama
Vector Database: ChromaDB (local persistent)
RAG Framework: LangChain
UI: Streamlit
Package Manager: UV
Language: Python 3.12

Problem Statement
==================

In industrial environments, maintenance engineers spend significant time manually searching through device manuals, fault code databases, and historical maintenance logs to diagnose machine anomalies. This process is slow, error-prone, and heavily dependent on individual expertise.

This project automates that entire diagnostic chain. When a sensor anomaly is detected, the system retrieves relevant knowledge from its vector database and generates a structured diagnosis report with likely cause, risk level, recommended action, and source reference.

Sample Queries
================
Scenario & Sample Query
==================================================
Motor vibration fault --- Motor-7 showing vibration 8.5 mm/s and temperature 88 degree 
                                C with grinding noise. Diagnose.
Compressor pressure --- Compressor-3 output pressure dropped from 175 PSI to 118 
                                PSI suddenly. What is the cause?
Historical lookup --- Has Motor-7 had any previous faults? What was the root 
                                cause and resolution?
Fault code lookup --- What does fault code E-47 mean and what action should I take?
Overload diagnosis --- Motor-12 drawing 40 percent more current than rated load for 
                                15 minutes. What could be the cause?

Diagnosis Report Format
=======================
For every query the system generates a structured response:

Likely Cause       : Description of probable fault root cause
Risk Level         : Low / Medium / High / Critical
Recommended Action : Step by step corrective action
Source Reference   : Document and page number used for the answer

Key Learning Outcomes
======================
End-to-end RAG pipeline implementation using LangChain
Local LLM inference using Ollama (no cloud API required)
Vector database setup and persistent storage with ChromaDB
Document chunking strategies and metadata tagging
Prompt engineering for structured diagnostic output
Streamlit UI for interactive RAG applications
Production-style project structure with centralised configuration

Author
=========
Kumar Gaurav
Senior Software Engineer | Python | GenAI | RAG | ETL Pipelines | IoT Data | Bengaluru, India# iot-rag-project
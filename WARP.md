# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Neuro-Compass** (semantic-KG) is an explainable AI agent framework for fMRI-based Alzheimer's Disease classification. The system implements a novel multi-agent architecture using Google's Agent Development Kit (ADK) to provide semantically rich, knowledge-grounded clinical explanations for neuroimaging AI models.

### 🎯 Core Mission
Solving the "black box" problem in neuroimaging AI by creating a trustworthy, autonomous AI assistant that transforms raw fMRI data into clinically relevant, explainable reports for neuroscientists.

## Key Architecture Components

### 🧠 Multi-Agent System (Google ADK)
- **Root Agent**: `agents/agent.py` - `fMRIAlzheimerPipeline` (SequentialAgent)
- **Stage 1 - Brain Mapping**: `agents/sub_agents/act_to_brain/` - Model inference and activation analysis
- **Stage 2 - Parallel Analysis**: `ExplainParallelAgent` orchestrates:
  - `image_explain`: Activation map medical interpretation 
  - `graph_rag`: Knowledge graph reasoning and entity linking
- **Stage 3 - Report Generation**: `agents/sub_agents/final_report/` - Clinical report synthesis

### 🕸️ Knowledge Graph Integration
- **Neo4j Database**: Brain region-function-symptom-disease relationships
- **GraphRAG System**: Entity linking + templated Cypher queries
- **Graph Files**: `graphql/` contains semantic knowledge graphs and visualizations
- **Query Client**: `agents/client/neo4j_client.py` for database interactions

### 🔬 Neural Network Models
- **CapsNet-RNN**: Primary model for AD classification in `scripts/capsnet/`
- **MCADNNet**: Alternative CNN architecture in `scripts/macadnnet/`
- **Dynamic Layer Selection**: LLM-driven "hypothesize-and-verify" explainability mechanism

## 🚀 Common Development Commands

### Environment Setup
```bash
# Install PyTorch with CUDA support (using light-the-torch)
python -m pip install light-the-torch
python -m light_the_torch install --upgrade torch torchaudio torchvision

# Or use poethepoet task
poetry run poe autoinstall-torch-cuda

# Install all dependencies
pip install -r requirements.txt
```

### 🧪 Model Training
```bash
# Train CapsNet-RNN model (primary model)
python -m scripts.capsnet.train

# Train MCADNNet model (alternative)
python -m scripts.macadnnet.train

# Prepare training data from raw fMRI files
python -m scripts.data_prepare
```

### 🔍 Single Model Inference
```bash
# CapsNet-RNN inference
python -m scripts.capsnet.infer

# MCADNNet inference with activation extraction
python -m scripts.macadnnet.inference \
    --model model/mcadnnet.pth \
    --input data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz
```

### 📊 Group-Level Analysis Pipeline
```bash
# Complete activation analysis workflow (run in order)
# 1. Generate activations for all subjects
python -m scripts.group.infer

# 2. Convert activation tensors to NIfTI format
python -m scripts.group.act_nii

# 3. Resample to standard atlas space
python -m scripts.group.resample

# 4. Generate brain region statistics
python -m scripts.group.brain_map

# 5. Optional: Get group average activations
python -m scripts.group.get_avg_act

# 6. Optional: Verify generated maps
python -m scripts.group.check_map
```

### 🕸️ Knowledge Graph Operations
```bash
# Build Neo4j graph database
python -m tools.build_neo4j

# Generate Cypher queries
python -m tools.generate_cypher

# Test knowledge graph connections
python -c "from agents.client.neo4j_client import Neo4jClient; client = Neo4jClient(); print('Neo4j connected')"
```

### 🤖 Multi-Agent Pipeline
```bash
# Run complete Neuro-Compass agent pipeline
python -m agents.agent

# Launch Streamlit interactive interface
streamlit run app.py

# Test backend analysis runner
python -m backend.backend_runner
```

### 🧪 Testing & Validation
```bash
# Test activation extraction
python -m tests.check_act

# Verify brain region mappings
python -m tests.brain_region

# Check model information
python -m tests.model_info

# Test individual components
python -m tests.image_explain
python -m tests.vertex  # Google ADK integration test
```

## 🏗️ Development Environment

### Required Environment Variables
Create a `.env` file in the root directory:
```bash
# Neo4j Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Google AI/LLM Services
GOOGLE_API_KEY=your_google_api_key

# Optional: For cloud deployment
PROJECT_ID=your_gcp_project_id
LOCATION=your_gcp_location
BUCKET_ID=your_gcp_bucket
```

### 📁 Data Directory Structure
```
data/
├── raw/                    # Original fMRI data
│   ├── AD/                # Alzheimer's patients
│   │   └── sub-*/         # Individual subjects
│   └── CN/                # Healthy controls
│       └── sub-*/         # Individual subjects
├── slices/                # Processed 2D slice images (if using slice-based training)
└── processed/             # Intermediate processing results

model/
├── capsnet/               # CapsNet-RNN model weights
│   └── best_capsnet_rnn.pth
└── macadnnet/            # MCADNNet model weights

output/                    # Analysis results
├── activations/           # Neural network activation maps
├── brain_maps/           # Brain region analysis results
└── visualizations/       # Generated plots and heatmaps

graphql/                  # Knowledge graph data
├── semantic_graph.graphml
├── nodes.csv
├── edges.csv
└── visualizations/
```

### 💻 Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for training/inference)
- **Memory**: 16GB+ RAM for processing fMRI data
- **Storage**: 50GB+ for datasets and model weights
- **Neo4j**: Running Neo4j database instance (local or remote)

### 🔧 System Dependencies
- Python 3.11+ (configured in pyproject.toml: `>=3.11,<3.14`)
- Neo4j database server
- CUDA toolkit (for GPU acceleration)
- Google Cloud credentials (for ADK agent services)

## 🧪 Testing & Validation Structure

### Integration Tests
- **Core Pipeline**: Run agents end-to-end with real data
- **Knowledge Graph**: Verify Neo4j connectivity and query results
- **Model Integration**: Test both CapsNet-RNN and MCADNNet

### Component Tests (`tests/` directory)
- `brain_region.py`: Brain atlas mapping validation
- `model_info.py`: Neural network architecture inspection
- `image_explain.py`: Activation explanation testing
- `vertex.py`: Google ADK integration validation
- `nii_check.py`: fMRI data loading and processing

## 📦 Technology Stack

### 🤖 AI/ML Framework
- **Agent Platform**: Google ADK (Agent Development Kit)
- **LLM**: Gemini 2.5 Flash Lite (via Google AI)
- **Deep Learning**: PyTorch, torchvision, torchinfo
- **Explainability**: grad-cam, custom activation analysis

### 🧠 Neuroimaging
- **Data Processing**: nibabel, nilearn, scikit-image
- **Visualization**: matplotlib, seaborn
- **Atlas**: AAL3 brain parcellation

### 🕸️ Knowledge Management
- **Graph Database**: Neo4j with Python driver
- **Graph Processing**: NetworkX for analysis
- **Data Export**: CSV, GraphML formats

### 🖥️ User Interface
- **Web App**: Streamlit for interactive analysis
- **Backend**: Custom async runner for agent orchestration
- **Visualization**: Interactive brain slice viewer

### 🛠️ Development Tools
- **Build System**: Poetry package manager
- **Task Runner**: poethepoet for automation
- **Code Quality**: Standard Python development practices

## 🚨 Common Issues & Solutions

### CUDA/GPU Issues
```bash
# If CUDA installation fails, use light-the-torch
python -m pip install light-the-torch
python -m light_the_torch install --upgrade torch torchaudio torchvision
```

### Neo4j Connection Issues
```bash
# Check Neo4j status
sudo systemctl status neo4j
# Restart Neo4j if needed
sudo systemctl restart neo4j
```

### Agent Execution Issues
- Ensure `.env` file contains valid Google API key
- Verify Neo4j database is running and accessible
- Check that required model weights are downloaded
- Ensure fMRI data is properly organized in `data/raw/`

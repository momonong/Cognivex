# Agent.md

This document provides comprehensive guidance for AI assistants working with the **Neuro-Compass** (semantic-KG) project - an explainable AI agent framework for fMRI-based Alzheimer's Disease classification.

## 🎯 Project Overview

**Neuro-Compass** is a cutting-edge explainable AI system that transforms fMRI neuroimaging data into clinically relevant, interpretable reports for Alzheimer's Disease diagnosis. The system solves the "black box" problem in medical AI by combining deep learning models with knowledge graph reasoning.

### Core Architecture
- **Multi-Agent System**: Built on Google ADK (Agent Development Kit)
- **Sequential Pipeline**: Brain mapping → Image analysis → Knowledge reasoning → Report generation
- **Knowledge Integration**: Neo4j graph database with semantic medical knowledge
- **Deep Learning**: CapsNet-RNN and MCADNNet models for AD classification

## 📁 Project Structure

```
semantic-KG/
├── agents/                    # Google ADK multi-agent system
│   ├── agent.py              # Root SequentialAgent (fMRIAlzheimerPipeline)
│   └── sub_agents/           # Individual pipeline stages
├── app/                      # Knowledge graph query utilities
├── app.py                    # Streamlit web interface
├── backend/                  # Async analysis runner
├── data/                     # fMRI datasets (AD/CN subjects)
├── graphql/                  # Neo4j knowledge graphs
├── model/                    # Trained neural network weights
├── scripts/                  # Data processing and training
├── tests/                    # Component validation tests
└── tools/                    # Neo4j and utility scripts
```

## 🚀 Key Development Commands

### Environment Setup
```bash
# Install PyTorch with CUDA support
python -m pip install light-the-torch
python -m light_the_torch install --upgrade torch torchaudio torchvision

# Or use poethepoet task
poetry run poe autoinstall-torch-cuda

# Install dependencies
pip install -r requirements.txt
```

### 🤖 Multi-Agent Pipeline
```bash
# Run complete analysis pipeline
python -m agents.agent

# Launch Streamlit interface
streamlit run app.py

# Run backend analysis (async/sync wrapper)
python -m backend.backend_runner
```

### 🧪 Model Training & Inference
```bash
# Train CapsNet-RNN (primary model)
python -m scripts.capsnet.train

# Train MCADNNet (alternative model)
python -m scripts.macadnnet.train

# Single model inference
python -m scripts.capsnet.infer
python -m scripts.macadnnet.inference --model model/mcadnnet.pth --input data/raw/AD/sub-14/...
```

### 🧠 Group Analysis Pipeline
```bash
# Complete activation analysis workflow (run in sequence)
python -m scripts.group.infer          # Generate activations
python -m scripts.group.act_nii        # Convert to NIfTI
python -m scripts.group.resample       # Atlas registration
python -m scripts.group.brain_map      # Region statistics
python -m scripts.group.get_avg_act    # Group averages (optional)
```

### 🕸️ Knowledge Graph Operations
```bash
# Build Neo4j database
python -m tools.build_neo4j

# Generate Cypher queries
python -m tools.generate_cypher

# Test connection
python -c "from agents.client.neo4j_client import Neo4jClient; client = Neo4jClient(); print('Neo4j connected')"
```

## 🔧 Technical Configuration

### Python Environment
- **Python Version**: 3.11+ (configured as `>=3.11,<3.14`)
- **Build System**: Poetry package manager
- **Task Runner**: poethepoet for automation

### Key Dependencies
```toml
# Core AI/ML
google-adk = ">=1.3.0,<2.0.0"
google-generativeai = ">=0.7.0,<0.9.0"
langgraph = ">=0.4.5,<0.5.0"
torch = "2.8.0"

# Neuroimaging
nibabel = ">=5.3.2,<6.0.0"
nilearn = ">=0.11.1,<0.12.0"
scikit-image = ">=0.25.2,<0.26.0"

# Knowledge Graph
neo4j = ">=5.28.1,<6.0.0"

# Web Interface
streamlit = ">=1.49.1,<2.0.0"

# Additional ML
groq = ">=0.31.1,<0.32.0"
litellm = ">=1.76.3,<2.0.0"
ollama = ">=0.5.3,<0.6.0"
```

### Required Environment Variables
```bash
# .env file setup
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_API_KEY=your_google_api_key

# Optional cloud deployment
PROJECT_ID=your_gcp_project_id
LOCATION=your_gcp_location
BUCKET_ID=your_gcp_bucket
```

## 🧩 Agent Architecture Details

### Root Agent (SequentialAgent)
- **Name**: `fMRIAlzheimerPipeline`
- **Function**: Orchestrates the complete analysis pipeline
- **Location**: `agents/agent.py`

### Sub-Agents Pipeline
1. **Brain Mapping**: `map_act_brain_agent` - Model inference and activation analysis
2. **Image Retrieval**: `retrieve_img_path_agent` - Locate visualization paths
3. **Image Explanation**: `image_explain_agent` - Medical interpretation of activation maps
4. **Knowledge Reasoning**: `graph_rag_agent` - Graph database query and entity linking
5. **Report Generation**: `report_generator_agent` - Clinical report synthesis

### Parallel Architecture (Currently Commented)
```python
# Available but currently disabled for stability
explain_parallel_agent = ParallelAgent(
    sub_agents=[graph_rag_agent, image_explain_agent]
)
```

## 🔬 Data Structure

### Input Data Format
- **fMRI Files**: NIfTI format (`.nii.gz`)
- **Directory Structure**: 
  - `data/raw/AD/sub-XX/` - Alzheimer's patients
  - `data/raw/CN/sub-XX/` - Healthy controls
- **Model Weights**: PyTorch format (`.pth`)
  - Primary: `model/capsnet/best_capsnet_rnn.pth`
  - Alternative: `model/macadnnet/`

### Output Formats
- **Activation Maps**: NIfTI format in atlas space
- **Visualizations**: PNG/JPG brain overlay images
- **Reports**: JSON structure with English/Chinese content
- **Brain Statistics**: CSV files with region-level metrics

## 🖥️ Web Interface Features

### Streamlit App (`app.py`)
- **Subject Selection**: Automatic detection from data directory
- **Model Selection**: CapsNet-RNN (primary), MCADNNet (alternative)
- **Interactive Viewer**: 3D brain slice exploration with nilearn
- **Real-time Analysis**: Async backend integration
- **Bilingual Reports**: English and Traditional Chinese output

### Key UI Components
```python
# Subject detection
subject_folders = glob.glob("data/raw/*/sub-*")

# Model execution
result_json = run_analysis_sync(subject_id, nii_path, model_path)

# Interactive brain viewer with sliders
plotting.plot_anat(img_3d, display_mode='x', cut_coords=[x])
```

## 🧪 Testing & Validation

### Component Tests (`tests/` directory)
```bash
# Core component validation
python -m tests.brain_region     # Brain atlas mapping
python -m tests.model_info        # Network architecture
python -m tests.image_explain     # Activation interpretation
python -m tests.vertex            # Google ADK integration
python -m tests.nii_check         # fMRI data loading
```

### Integration Testing
- **Full Pipeline**: End-to-end agent execution with real data
- **Knowledge Graph**: Neo4j connectivity and query validation
- **Model Integration**: Both CapsNet-RNN and MCADNNet testing

## 🚨 Common Issues & Solutions

### CUDA/GPU Setup
```bash
# Recommended installation method
python -m pip install light-the-torch
python -m light_the_torch install --upgrade torch torchaudio torchvision
```

### Neo4j Database
```bash
# Service management
sudo systemctl status neo4j
sudo systemctl restart neo4j
```

### Agent Execution Requirements
- Valid Google API key in `.env`
- Running Neo4j database instance
- Model weights downloaded and accessible
- Proper fMRI data organization in `data/raw/`

## 📈 Recent Development Progress

### Current Branch: `develop/optimize-stability`
- **Latest Commits**: 
  - f4cc844: Image path passing optimization attempts
  - 259e929: Model switching flexibility improvements
  - 7e16d4e: Stability optimization trials
  - cedad93: Classification result passing method updates

### Key Recent Changes
- **Vertex AI Integration**: Migrated to Google Vertex AI services
- **Streamlit Optimization**: Enhanced web interface stability
- **Model Flexibility**: Improved switching between CapsNet-RNN and MCADNNet
- **Error Handling**: Better error messages and user feedback

## 🎯 Development Guidelines for AI Assistants

### When Working with This Project:
1. **Always check environment setup** before running commands
2. **Verify Neo4j connectivity** before agent execution
3. **Use appropriate model paths** based on selected architecture
4. **Check data directory structure** matches expected format
5. **Monitor GPU memory usage** during training/inference
6. **Use poetry/poethepoet** for dependency management
7. **Test components individually** before full pipeline runs

### Common Debugging Steps:
1. Check `.env` file configuration
2. Verify Neo4j service status
3. Confirm model weight file existence
4. Validate fMRI data file paths
5. Test Google ADK authentication
6. Monitor system resource usage

### File Path Patterns:
- Models: `model/capsnet/best_capsnet_rnn.pth`
- Data: `data/raw/AD/sub-14/dswausub-098_S_6601_task-rest_bold.nii.gz`
- Output: `output/activations/`, `output/visualizations/`
- Knowledge: `graphql/semantic_graph.graphml`

This framework represents a state-of-the-art approach to explainable AI in medical neuroimaging, combining the latest in multi-agent architectures with domain-specific knowledge integration.
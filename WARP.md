# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**semantic-KG** is an Alzheimer's disease neuroimaging analysis pipeline that combines deep learning models with knowledge graph reasoning. The system processes fMRI data through a multi-agent pipeline to detect AD patterns and provide semantic explanations using brain region knowledge graphs.

## Key Architecture Components

### Agent-Based Pipeline
- **Root Agent**: `agents/agent.py` - Sequential multi-step pipeline coordinator
- **Sub-agents**:
  - `nii_inference`: NIfTI fMRI data processing and CNN inference
  - `image_explain`: GradCAM-based activation explanation
  - `graph_rag`: Knowledge graph reasoning and retrieval
  - `final_report`: Clinical report generation

### Knowledge Graph System
- **Neo4j Integration**: Brain region-function-symptom-disease relationships
- **Graph Files**: `graphql/` contains semantic graph data and visualizations
- **Query System**: `app/kg_query.py` for Neo4j database interactions

### Neural Network Models
- **MCADNNet**: Main CNN architecture for AD classification in `scripts/macadnnet/`
- **CapsNet**: Alternative capsule network implementation in `scripts/capsnet/`
- **Training Scripts**: Located in respective model directories

## Common Development Commands

### Environment Setup
```bash
# Install PyTorch with CUDA support
poetry run poe autoinstall-torch-cuda

# Install dependencies
poetry install
```

### Data Pipeline
```bash
# Prepare fMRI slice data (requires raw data in data/raw/)
poetry run python -m scripts.data_prepare

# Train models
poetry run python -m scripts.macadnnet.train
poetry run python -m scripts.capsnet.train
```

### Inference Pipeline
```bash
# Single image inference with activation extraction
poetry run python -m scripts.macadnnet.inference \
    --image data/images/AD/AD_sub-AD_027_S_6648_task-rest_bold.nii_z001_t003.png \
    --weights model/mcadnnet_mps.pth \
    --extract-activation \
    --activation-output activation
```

### Activation Analysis Workflow
```bash
# Group-level activation analysis sequence
poetry run python -m scripts.group.infer
poetry run python -m scripts.group.act_nii
poetry run python -m scripts.group.resample
poetry run python -m scripts.group.get_avg_act  # optional
poetry run python -m scripts.group.brain_map
poetry run python -m scripts.group.check_map
```

### Knowledge Graph Operations
```bash
# Build and visualize semantic knowledge graph
poetry run python test_overall.py

# Generate Neo4j import files
poetry run python -m tools.build_neo4j
```

### Agent Pipeline Testing
```bash
# Run complete multi-agent pipeline
poetry run python -m agents.agent

# Test overall pipeline with knowledge graph integration
poetry run python test_overall.py
```

### Cloud Deployment
```bash
# Deploy agent to Google Cloud Vertex AI
poetry run python -m deployment.deploy --create --project_id=<PROJECT_ID> --location=<LOCATION> --bucket=<BUCKET>

# List deployed agents
poetry run python -m deployment.deploy --list

# Delete deployed agent
poetry run python -m deployment.deploy --delete --resource_id=<RESOURCE_ID>
```

## Development Environment Requirements

### Required Environment Variables
Create a `.env` file with:
```
NEO4J_PASSWORD=your_neo4j_password
PROJECT_ID=your_gcp_project_id
LOCATION=your_gcp_location
BUCKET_ID=your_gcp_bucket
```

### Data Directory Structure
- `data/raw/`: Original fMRI data (AD/ and CN/ subdirectories)
- `data/slices/`: Processed 2D slice images organized by fold
- `model/`: Trained model weights
- `output/`: Inference results and activations
- `graphql/`: Knowledge graph files and visualizations

### Hardware Considerations
- GPU recommended for training (CUDA support via light-the-torch)
- Neo4j database for knowledge graph operations
- Google Cloud credentials for deployment (service account JSON)

## Testing Structure

- `test_overall.py`: Integration test for the complete pipeline
- `tests/`: Various component-specific tests and utilities
- Individual model directories contain their own test scripts

## Key Dependencies

- **Deep Learning**: PyTorch, torchvision, grad-cam
- **Neuroimaging**: nibabel, nilearn, scikit-image
- **Knowledge Graph**: neo4j, networkx
- **Cloud AI**: google-cloud-aiplatform, google-adk
- **Data Processing**: numpy, pandas, opencv-python
- **Build System**: Poetry with poethepoet tasks

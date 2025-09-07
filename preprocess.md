# Act-to-Brain Preprocessing Pipeline

This document describes the detailed preprocessing pipeline implemented in the `MapActBrainAgent` for fMRI Alzheimer's disease analysis.

```mermaid
flowchart TD
    %% Main Pipeline Flow
    Input(["📄 fMRI Input<br/>.nii.gz + .pth model"]) --> ModelAnalysis["🔍 Model Analysis<br/>• Inspect layers<br/>• LLM layer selection<br/>• Validate layers"]
    
    ModelAnalysis --> ModelPrep["🔧 Model Preparation<br/>• Attach hooks<br/>• Load weights"]
    
    ModelPrep --> Inference["🧠 NIfTI Inference<br/>• Sliding window<br/>• Forward pass<br/>• Save activations"]
    
    Inference --> Classification["📊 Classification<br/>AD or CN"]
    
    Inference --> Filtering["🎛️ Dynamic Filtering<br/>LLM-based layer selection"]
    
    Filtering --> ProcessLayers["🔄 Layer Processing<br/>(For each selected layer)"]
    
    %% Layer Processing Subgraph
    subgraph LayerLoop ["Layer Processing Loop"]
        direction TB
        A["🧩 Activation → NIfTI<br/>Select channel & interpolate"] 
        B["🎯 Resample to Atlas<br/>AAL3 brain atlas alignment"]
        C["🗺️ Brain Region Analysis<br/>Map to 166 brain regions"]
        D["📊 Visualization<br/>Generate 3D heatmap"]
        
        A --> B --> C --> D
    end
    
    ProcessLayers --> LayerLoop
    LayerLoop --> Results["📋 Final Results<br/>JSON with brain analysis"]
    
    %% Key Data Transformations
    subgraph DataFlow ["Data Transformations"]
        direction LR
        fMRI["fMRI<br/>[X,Y,Z,T]"] --> Tensor["Activation<br/>[C,D,H,W]"] 
        Tensor --> NIfTI["NIfTI<br/>[X,Y,Z]"] 
        NIfTI --> Atlas["Atlas Space<br/>AAL3 aligned"]
        Atlas --> Regions["Region Stats<br/>166 brain areas"]
    end
    
    %% Styling
    classDef mainFlow fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef data fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    
    class Input,ModelAnalysis,ModelPrep,Inference,Filtering,ProcessLayers,Results mainFlow
    class Classification,A,B,C,D processing
    class fMRI,Tensor,NIfTI,Atlas,Regions data
```

## Core Pipeline Steps

### 🔍 Model Analysis
Combines model structure inspection, LLM-based layer selection, and validation:
- Analyzes CapsNet-RNN architecture using `torchsummary`
- LLM intelligently selects spatial feature layers (Conv3d, capsules)
- Validates layer existence to prevent runtime errors

### 🔧 Model Preparation
Prepares model for activation capture:
- Attaches forward hooks to selected layers
- Loads pre-trained weights and sets evaluation mode
- Configures optimal device (CUDA/MPS/CPU)

### 🧠 NIfTI Inference
Processes fMRI data through the neural network:
- **Input**: 4D fMRI [X, Y, Z, T] → sliding windows for temporal modeling
- **Processing**: Forward pass with activation capture via hooks
- **Output**: Classification result (AD/CN) + saved activation tensors

### 🎛️ Dynamic Filtering
LLM-based quality control:
- Analyzes activation statistics for each captured layer
- Selects most informative layers for further processing
- Removes layers with poor activation patterns

### 🔄 Layer Processing Loop
For each selected layer, sequential processing:

1. **🧩 Activation → NIfTI**: Select strongest channel, interpolate to fMRI dimensions
2. **🎯 Atlas Alignment**: Resample to AAL3 brain atlas coordinate system
3. **🗺️ Brain Mapping**: Map activations to 166 brain regions with statistics
4. **📊 Visualization**: Generate 3D brain heatmaps and save PNG outputs

## Data Transformation Flow

The pipeline transforms fMRI data through these key stages:

```
fMRI [X,Y,Z,T] → Sliding Windows → Neural Activations [C,D,H,W] → 
Strongest Channel [D,H,W] → Interpolated NIfTI [X,Y,Z] → 
AAL3 Atlas Space → 166 Brain Region Statistics → 3D Visualization
```

## Key Features

- **🤖 LLM-Guided Processing**: Intelligent layer selection and quality control
- **🧠 Neuroanatomical Mapping**: AAL3 atlas with 166 brain regions
- **⚡ Optimized Pipeline**: Sliding window + strongest channel selection
- **🎯 Clinical Focus**: AD/CN classification with explainable activations

## Technical Specs

- **Model**: CapsNet-RNN (3D Conv + Capsule layers)
- **Atlas**: AAL3v1 (166 regions, MNI space)
- **Processing**: Window=5, Stride=3, 99th percentile thresholding
- **Output**: JSON with classification, brain regions, and visualization paths

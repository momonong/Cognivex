# Neuro-Compass: An Explainable AI Agent for Neuroimaging

This project introduces **Neuro-Compass**, a novel, autonomous multi-agent framework designed to provide robust, semantically rich explanations for fMRI-based Alzheimer's Disease (AD) classification models.

The core challenge in applying AI to neuroimaging is the "black box" problem. While models can achieve high accuracy, their lack of transparent reasoning hinders clinical adoption and trust. Neuro-Compass is engineered to solve this by creating a trustworthy AI assistant for neuroscientists. It automates the entire analysis pipeline, transforming raw fMRI data into a clinically relevant, knowledge-grounded report.

### Key Features:
* **🧠 Autonomous Multi-Agent System**: Orchestrates a team of specialist AI agents (built with Google's ADK) to manage a complex, multi-stage workflow.
* **🔍 Dynamic XAI Layer Selection**: Implements a "hypothesize-and-verify" mechanism where an LLM first proposes, and then validates against real activation data, the most informative neural network layer for explainability.
* **🔗 Robust Knowledge Graph Integration (GraphRAG)**: Solves the entity linking challenge by using an LLM to align model-identified brain regions with a canonical list from a Neo4j graph, followed by a 100% reliable templated Cypher query.
* **📄 Publication-Ready Reporting**: The final agent synthesizes all computational results, visual interpretations, and knowledge graph insights into a comprehensive, structured clinical report.
* **🔬 Scientifically-Grounded**: Our case studies show the system's ability to autonomously identify activation patterns in the Default Mode Network (DMN), a key neuropathological correlate of AD, validating the model's clinical relevance.

---

## 🚀 Getting Started

This guide will walk you through setting up the project and running the complete analysis pipeline.

### 1. Prerequisites

* Python 3.10+
* An NVIDIA GPU with CUDA (recommended for training and inference)
* A running Neo4j database instance

### 2. Installation

We recommend using a virtual environment to manage dependencies.

**Step 1: Clone the repository**

**Step 2: Create and activate a virtual environment**
```bash
# For Unix/macOS
python3 -m venv .venv
source .venv/bin/activate

# For Windows
python -m venv .venv
.venv\Scripts\activate
```

**Step 3: Install the required packages**
```bash
pip install -r requirements.txt
```

**Step 4: Set up environment variables**
Create a `.env` file in the root directory and add your credentials:
```
# .env file
GOOGLE_API_KEY="your_google_api_key"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_neo4j_password"
```

Step 5: Set Up the Knowledge Graph
This project relies on a Neo4j knowledge graph. Please ensure your Neo4j database is running. To build the required graph schema and populate it with data, please follow the step-by-step instructions in the document below:

➡️ Knowledge Graph Setup Guide refers to `scripts/kg/build_kg.md` in this repo.

### 3. Usage

The project is divided into several key stages.

**Stage 1: Data Preparation**

This project requires a specific fMRI dataset for analysis. Please follow these steps to download and set it up correctly.

**Step 1: Download the Dataset**
Download the compressed raw fMRI dataset from the cloud storage link provided below. The file will be a `.zip` or `.tar.gz` archive.

[**➡️ Download Raw fMRI Dataset Here**](https://u.pcloud.link/publink/show?code=kZEgL15ZhlezDWqfUEY3MkFwUK9Gtui7w0T7)  

**Step 2: Unzip and Place the Data**
1.  Unzip the file you just downloaded (e.g., `data.zip`).
2.  You should get a folder containing the subject data in `raw` sub-folder (e.g., `AD/` and `CN/` sub-folders).
3.  In the root directory of this project, create a folder named `data/` if it doesn't exist.
4.  Place the unzipped contents into a new folder named `raw` inside the `data/` directory.

**Stage 2: Model Weights & Training**

This project provides pre-trained model weights for immediate inference and explainability analysis. Alternatively, advanced users can train the model from scratch on their own data.

### Using Pre-trained Weights (Recommended for Demo)

We provide the weights for the `CapsNetRNN` model used in our case study. You can download them from the link below:

[**➡️ Download Pre-trained Model Weights Here**](https://u.pcloud.link/publink/show?code=kZ7gL15ZoCYrxwMqwwQmmBYDWfDmuy2GB4Ly)

After downloading, please place the model weights file (e.g., `best_capsnet_rnn.pth`) into the `model/capsnet/` directory.

**Important Note on Accuracy:**
Please be aware that these weights were trained on a limited dataset solely for the purpose of demonstrating this explainability framework. As such, the model's predictive accuracy is not at a clinical-grade level. The primary focus of this project is the **agent-based explainability framework itself**, which is model-agnostic and can be applied to any higher-accuracy model in the future.

### Training from Scratch (Optional)

If you wish to train the model on your own dataset, you can run the training script. Ensure your data has been prepared correctly in Stage 1.
```shell
python -m scripts.train
```

**Stage 3: Activation Analysis Pipeline**
This sequence of scripts performs the core explainability analysis, from inference to final brain map generation. Run them in the specified order.

```shell
# 1. Run inference on all subjects to generate activations
python -m scripts.group.infer

# 2. Convert activation tensors to NIfTI format
python -m scripts.group.act_nii

# 3. Resample activation maps to the standard atlas space
python -m scripts.group.resample

# 4. (Optional) Calculate the group-average activation map
python -m scripts.group.get_avg_act

# 5. Generate quantitative brain region statistics
python -m scripts.group.brain_map

# 6. (Optional) Check the generated maps
python -m scripts.group.check_map
```

**Stage 4: Launch the Agent-based UI**
To interact with the final multi-agent system and view the generated reports, launch the Streamlit application:
```shell
streamlit run app.py
```

---

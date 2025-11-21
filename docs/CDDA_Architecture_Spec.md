# Cognitive Discrepancy-Driven Agent (CDDA) Framework
## Architecture Specification for Cognivex Project

**Version:** 1.0  
**Date:** November 19, 2025  
**Purpose:** Master's Thesis - Formalized Architecture Documentation

---

## Executive Summary

The Cognitive Discrepancy-Driven Agent (CDDA) Framework is a five-layer intelligent system designed for Alzheimer's disease diagnosis support. The framework combines machine learning predictions with uncertainty quantification, autonomous agent reasoning, and knowledge graph integration to provide explainable, trustworthy diagnostic insights.

---

## 1. System Architecture: The 5 Layers (Updated for Phase 4)

### Layer 1: Tool Kit (RF/SHAP)
**Purpose:** Core ML prediction and explainability engine

- **Components:**
  - Random Forest (RF) classifier for AD/NC/MCI classification
  - SHAP (SHapley Additive exPlanations) for feature importance
  - ROI-based feature extraction from structural MRI
  - Gray matter volume analysis

- **Responsibilities:**
  - Execute diagnostic predictions on patient neuroimaging data
  - Generate local explainability metrics (SHAP values per feature)
  - Extract top contributing brain regions (ROIs)
  - Provide raw prediction probabilities

### Layer 2: Trust/Calibration (UQ/Z-Score Logic)
**Purpose:** Uncertainty quantification and anomaly detection

- **Components:**
  - Uncertainty Quantification (UQ) scoring system
  - Z-score normalization for feature deviation analysis
  - Anomaly detection for out-of-distribution patterns
  - Confidence calibration metrics

- **Responsibilities:**
  - Calculate uncertainty scores for each prediction
  - Identify statistically significant feature deviations (z-scores)
  - Flag anomalous brain regions (e.g., SN_pc substantia nigra)
  - Provide trust indicators for downstream reasoning

### Layer 3: Cognitive/Orchestration (A2A Dual-LLM Architecture) **[PHASE 4 UPDATE]**
**Purpose:** Autonomous decision-making and clinical reasoning with dual-LLM architecture

**Phase 4 introduces the Agent-to-Agent (A2A) pattern with Model Context Protocol (MCP):**

#### Context Layer: MCP Server
- **Purpose:** Separate read-only data (Resources) from executable actions (Tools)
- **Components:**
  - DiagnosticMCPServer implementing MCP protocol
  - Resource handlers (diagnosis://, knowledge://)
  - Tool handlers (simulate_counterfactual)
- **Responsibilities:**
  - Provide MCP-compliant interface for resources and tools
  - Route resource URIs to appropriate backends
  - Execute tools with validated arguments
  - Handle errors gracefully with fallback mechanisms

#### Cognitive Layer: A2A Agent System
- **Agent A (Orchestrator):** GPT-OSS-20B or similar function-calling model
  - **Role:** MCP client, reads resources, invokes tools, compiles context
  - **Responsibilities:**
    - Read diagnostic reports from MCP server
    - Evaluate signals (UQ score, anomaly status)
    - Decide which tools to invoke based on signals
    - Compile ContextObject for handoff to Agent B
    - Log all decisions with reasoning chains
  
- **Agent B (Clinical Consultant):** MedGemma-27B or similar medical domain model
  - **Role:** Medical specialist, synthesizes clinical reports
  - **Responsibilities:**
    - Receive ContextObject from Agent A (NO direct tool access)
    - Synthesize clinical narratives from provided context
    - Interpret counterfactual results in clinical terms
    - Flag potential mixed pathology and anomalies
    - Generate evidence-based recommendations

**Key A2A Principles:**
- Clear separation: Agent A handles orchestration, Agent B handles clinical reasoning
- Handoff via ContextObject ensures Agent B has no direct tool access
- Complete reasoning chain from both agents for transparency
- Graceful fallback to rule-based logic if LLMs unavailable

### Layer 4: Knowledge (GraphRAG)
**Purpose:** Domain knowledge retrieval and contextual reasoning

- **Components:**
  - Neo4j knowledge graph
  - Alzheimer's disease ontology (brain regions, symptoms, biomarkers)
  - GraphRAG retrieval system
  - Entity linking for ROI-to-knowledge mapping

- **Responsibilities:**
  - Retrieve relevant medical knowledge for flagged anomalies
  - Provide contextual explanations for brain region findings
  - Link ROI features to clinical significance
  - Support evidence-based reasoning
  - **[Phase 4]** Fallback to local knowledge base if Neo4j unavailable

### Layer 5: Presentation (Streamlit/UI)
**Purpose:** User interface and visualization

- **Components:**
  - Streamlit web application
  - Interactive brain region visualizations
  - Diagnostic report rendering
  - Counterfactual simulation interface
  - **[Phase 4]** Reasoning chain display for transparency

- **Responsibilities:**
  - Display diagnostic predictions and confidence scores
  - Visualize SHAP values and feature importance
  - Present agent reasoning chains with timestamps
  - Enable "What-If" scenario exploration
  - Show MCP actions and A2A handoff details

---

## 1.1 Phase 4: MCP and A2A Architecture Details

### Model Context Protocol (MCP)

The MCP server provides a standardized interface for accessing diagnostic resources and tools:

**Resources (Read-Only Data):**
```
diagnosis://{subject_id}/report    - Complete diagnostic data
diagnosis://{subject_id}/features  - Raw feature values
knowledge://{region_name}/context  - Clinical knowledge from graph
```

**Tools (Executable Actions):**
```
simulate_counterfactual(subject_id, features_to_mask) - What-if analysis
```

**MCP Benefits:**
- Clean separation between data access and action execution
- Standardized URI-based resource access
- Validated tool invocation with JSON schemas
- Error handling with graceful degradation

### Agent-to-Agent (A2A) Handoff Protocol

The A2A pattern ensures clear separation of concerns between orchestration and clinical reasoning:

**Handoff Flow:**
```
1. Agent A (Orchestrator):
   - Reads: diagnosis://{subject_id}/report
   - Evaluates: UQ score, anomaly status
   - Decides: Which tools to invoke (if any)
   - Compiles: ContextObject with all gathered data
   
2. Handoff: Agent A → Agent B
   - ContextObject contains: diagnostic_report, tool_results, signals, reasoning
   - Agent B receives complete context (NO direct MCP access)
   
3. Agent B (Consultant):
   - Receives: ContextObject from Agent A
   - Synthesizes: Clinical narrative from provided context
   - Returns: Final diagnostic report with recommendations
```

**ContextObject Structure:**
```python
{
  "subject_id": str,
  "diagnostic_report": DiagnosticReport,
  "tool_results": Optional[Dict],  # counterfactual or knowledge_context
  "decision_rationale": str,
  "signals": Dict,  # uq_score, has_anomaly, etc.
  "agent_a_reasoning": List[str],
  "mcp_actions": List[MCPAction],
  "timestamp": str
}
```

**A2A Benefits:**
- Agent A focuses on data gathering and tool orchestration
- Agent B focuses purely on clinical reasoning and synthesis
- Clear handoff ensures Agent B has no tool access (prevents confusion)
- Complete reasoning chain from both agents for transparency
- Fallback mechanisms at each layer for robustness

---

## 2. Core Tool Definitions: The API

### Tool 1: `get_diagnostic_report(subject_id)`

**Purpose:** Provide all factual and contextual data for the Agent to reason over.

**Input:**
- `subject_id` (str): Unique patient identifier

**Mandatory Output Fields:**

```python
{
    "subject_id": str,
    "prediction_result": str,  # "AD", "NC", or "MCI"
    "confidence": float,  # 0.0 to 1.0
    "uq_score": float,  # Uncertainty quantification score (0.0 to 1.0)
    "top_features": [
        {
            "roi_name": str,  # Brain region name (e.g., "Hippocampus_L")
            "feature_value": float,  # Raw feature value
            "z_score": float,  # Standardized deviation from population mean
            "shap_value": float,  # SHAP contribution to prediction
            "rank": int  # Importance ranking (1 = most important)
        }
    ],
    "anomaly_status": {
        "has_anomaly": bool,
        "anomalous_regions": [str],  # List of flagged ROI names (e.g., ["SN_pc"])
        "anomaly_type": str  # "statistical_outlier", "rare_pattern", etc.
    },
    "metadata": {
        "model_version": str,
        "timestamp": str
    }
}
```

**Example Output:**
```json
{
    "subject_id": "sub-0005",
    "prediction_result": "AD",
    "confidence": 0.87,
    "uq_score": 0.82,
    "top_features": [
        {
            "roi_name": "Hippocampus_L",
            "feature_value": 2345.6,
            "z_score": -2.8,
            "shap_value": 0.15,
            "rank": 1
        },
        {
            "roi_name": "SN_pc",
            "feature_value": 189.2,
            "z_score": -3.5,
            "shap_value": 0.12,
            "rank": 2
        }
    ],
    "anomaly_status": {
        "has_anomaly": true,
        "anomalous_regions": ["SN_pc"],
        "anomaly_type": "statistical_outlier"
    }
}
```

---

### Tool 2: `simulate_counterfactual(subject_id, features_to_mask)`

**Purpose:** Execute a "What-If" prediction experiment by masking/adjusting specific features.

**Input:**
- `subject_id` (str): Unique patient identifier
- `features_to_mask` (list[str]): ROI names to neutralize or adjust

**Mandatory Output Fields:**

```python
{
    "subject_id": str,
    "original_prediction": str,
    "original_confidence": float,
    "new_prediction": str,
    "new_confidence": float,
    "confidence_delta": float,  # new_confidence - original_confidence
    "masked_features": [
        {
            "roi_name": str,
            "original_value": float,
            "masked_value": float,  # Typically population mean
            "impact": float  # Contribution to confidence change
        }
    ],
    "interpretation": str  # Agent-generated natural language summary
}
```

**Example Output:**
```json
{
    "subject_id": "sub-0005",
    "original_prediction": "AD",
    "original_confidence": 0.87,
    "new_prediction": "AD",
    "new_confidence": 0.72,
    "confidence_delta": -0.15,
    "masked_features": [
        {
            "roi_name": "Hippocampus_L",
            "original_value": 2345.6,
            "masked_value": 3200.0,
            "impact": -0.15
        }
    ],
    "interpretation": "Masking hippocampal atrophy reduces AD confidence by 15%, indicating this region is a primary driver of the diagnosis."
}
```

---

## 3. Agent Decision Logic: The CDDA Flowchart

### High-Level Autonomous Decision Flow

The Layer 3 Agent operates autonomously based on signals from Layer 2 (Trust/Calibration). This decision logic is what differentiates CDDA from traditional ML pipelines.

```
┌─────────────────────────────────────────────────────────────┐
│  START: Agent receives diagnostic request for subject_id   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  STEP 1: Call Tool 1              │
         │  get_diagnostic_report(subject_id)│
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  STEP 2: Parse Output             │
         │  - prediction_result              │
         │  - confidence                     │
         │  - uq_score                       │
         │  - anomaly_status                 │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  DECISION POINT 1: UQ Check       │
         │  IF uq_score > 0.8 (high UQ)      │
         └───────┬───────────────────┬───────┘
                 │ YES               │ NO
                 ▼                   │
    ┌────────────────────────┐      │
    │  ACTION 1:             │      │
    │  Call Tool 2           │      │
    │  simulate_counterfactual│     │
    │  (top 3 features)      │      │
    └────────────┬───────────┘      │
                 │                   │
                 ▼                   │
    ┌────────────────────────┐      │
    │  Generate explanation: │      │
    │  "High uncertainty     │      │
    │  detected. Simulation  │      │
    │  shows X feature       │      │
    │  drives Y% of result." │      │
    └────────────┬───────────┘      │
                 │                   │
                 └───────┬───────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  DECISION POINT 2: Anomaly Check  │
         │  IF anomaly_status.has_anomaly    │
         └───────┬───────────────────┬───────┘
                 │ YES               │ NO
                 ▼                   │
    ┌────────────────────────┐      │
    │  ACTION 2:             │      │
    │  Call Tool 4 (GraphRAG)│      │
    │  Lookup anomalous ROIs │      │
    │  (e.g., SN_pc)         │      │
    └────────────┬───────────┘      │
                 │                   │
                 ▼                   │
    ┌────────────────────────┐      │
    │  Generate explanation: │      │
    │  "Substantia nigra     │      │
    │  atrophy detected.     │      │
    │  Clinical significance:│      │
    │  [KG context]"         │      │
    └────────────┬───────────┘      │
                 │                   │
                 └───────┬───────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  ELSE: Standard Report            │
         │  Generate baseline diagnostic     │
         │  report with SHAP explanations    │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  STEP 3: Synthesize Final Report  │
         │  - Prediction + confidence        │
         │  - Top features + SHAP values     │
         │  - Counterfactual insights (if any)│
         │  - Knowledge context (if any)     │
         │  - Natural language summary       │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │  END: Return to Layer 5 (UI)      │
         └───────────────────────────────────┘
```

### Decision Logic Pseudocode

```python
def cdda_agent_workflow(subject_id):
    # STEP 1: Get diagnostic data
    report = get_diagnostic_report(subject_id)
    
    # Initialize response
    response = {
        "prediction": report["prediction_result"],
        "confidence": report["confidence"],
        "explanations": []
    }
    
    # DECISION POINT 1: High Uncertainty Check
    if report["uq_score"] > 0.8:
        # ACTION 1: Run counterfactual simulation
        top_features = [f["roi_name"] for f in report["top_features"][:3]]
        simulation = simulate_counterfactual(subject_id, top_features)
        
        response["explanations"].append({
            "type": "counterfactual",
            "message": f"High uncertainty detected (UQ={report['uq_score']:.2f}). "
                      f"Simulation shows {simulation['confidence_delta']:.2%} "
                      f"confidence change when masking key features.",
            "data": simulation
        })
    
    # DECISION POINT 2: Anomaly Detection Check
    if report["anomaly_status"]["has_anomaly"]:
        # ACTION 2: Query knowledge graph
        anomalous_rois = report["anomaly_status"]["anomalous_regions"]
        kg_context = query_knowledge_graph(anomalous_rois)
        
        response["explanations"].append({
            "type": "knowledge_context",
            "message": f"Anomalous pattern detected in {', '.join(anomalous_rois)}. "
                      f"Clinical significance: {kg_context['summary']}",
            "data": kg_context
        })
    
    # STEP 3: Generate natural language summary
    response["summary"] = generate_nl_summary(report, response["explanations"])
    
    return response
```

### Key Decision Thresholds

| Parameter | Threshold | Action Triggered |
|-----------|-----------|------------------|
| `uq_score` | > 0.8 | Call Tool 2 (Counterfactual Simulation) |
| `anomaly_status.has_anomaly` | `true` | Call Tool 4 (GraphRAG Lookup) |
| `z_score` | < -2.5 or > 2.5 | Flag as anomalous feature |
| `confidence` | < 0.6 | Recommend additional clinical evaluation |

---

## 4. Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Layer 5: Presentation                     │
│                   (Streamlit UI)                             │
└────────────────────────┬─────────────────────────────────────┘
                         │ User Request
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                Layer 3: Cognitive Agent                      │
│              (LangChain + LLM Orchestration)                 │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Tool Caller  │  │ Decision     │  │ NL Generator │      │
│  │              │  │ Logic        │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────┬─────────────────────┬─────────────────────┬────────────┘
     │                     │                     │
     │ Tool 1              │ Tool 2              │ Tool 4
     ▼                     ▼                     ▼
┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  Layer 1:   │  │    Layer 2:     │  │    Layer 4:      │
│  Tool Kit   │  │ Trust/Calib     │  │   Knowledge      │
│             │  │                 │  │                  │
│ ┌─────────┐ │  │ ┌─────────────┐ │  │ ┌──────────────┐ │
│ │ RF Model│ │  │ │ UQ Scorer   │ │  │ │ Neo4j Graph  │ │
│ └─────────┘ │  │ └─────────────┘ │  │ └──────────────┘ │
│ ┌─────────┐ │  │ ┌─────────────┐ │  │ ┌──────────────┐ │
│ │ SHAP    │ │  │ │ Z-Score     │ │  │ │ Entity Linker│ │
│ └─────────┘ │  │ └─────────────┘ │  │ └──────────────┘ │
│ ┌─────────┐ │  │ ┌─────────────┐ │  │ ┌──────────────┐ │
│ │ ROI Ext │ │  │ │ Anomaly Det │ │  │ │ Query Engine │ │
│ └─────────┘ │  │ └─────────────┘ │  │ └──────────────┘ │
└─────────────┘  └─────────────────┘  └──────────────────┘
```

---

## 5. Implementation Roadmap

### Phase 1: Tool Kit Foundation (Layer 1 + Layer 2)
- Implement `get_diagnostic_report()` function
- Integrate existing RF model and SHAP explainer
- Add UQ scoring logic
- Implement z-score calculation
- Add anomaly detection for statistical outliers

### Phase 2: Agent Orchestration (Layer 3)
- Set up LangChain agent framework
- Implement tool-calling interface
- Code CDDA decision logic (IF-THEN rules)
- Implement `simulate_counterfactual()` function
- Add natural language generation

### Phase 3: Knowledge Integration (Layer 4)
- Connect to Neo4j knowledge graph
- Implement GraphRAG retrieval
- Add entity linking for ROI-to-knowledge mapping
- Create query templates for common anomalies

### Phase 4: UI Integration (Layer 5)
- Build Streamlit diagnostic dashboard
- Add interactive visualizations
- Implement counterfactual simulation UI
- Add agent reasoning chain display

---

## 6. Key Innovations of CDDA

1. **Uncertainty-Driven Reasoning:** Unlike traditional ML systems that only provide predictions, CDDA uses uncertainty signals to trigger deeper analysis.

2. **Autonomous Tool Orchestration:** The agent autonomously decides which tools to call based on data characteristics, not pre-programmed workflows.

3. **Counterfactual Explainability:** Goes beyond feature importance to show "what would change if X were different."

4. **Knowledge-Grounded Anomaly Handling:** Anomalous patterns trigger knowledge graph lookups for clinical context.

5. **Transparent Decision Logic:** All agent decisions are traceable and explainable through the CDDA flowchart.

---

## 7. Evaluation Metrics

### Layer 1 (Tool Kit)
- Model accuracy, precision, recall, F1-score
- SHAP value consistency

### Layer 2 (Trust/Calibration)
- UQ score correlation with prediction errors
- Anomaly detection precision/recall

### Layer 3 (Cognitive Agent)
- Tool selection accuracy (correct tool for scenario)
- Explanation quality (human evaluation)
- Reasoning chain coherence

### Layer 4 (Knowledge)
- Retrieval relevance (precision@k)
- Entity linking accuracy

### Layer 5 (Presentation)
- User satisfaction scores
- Task completion time
- Diagnostic confidence improvement

---

## 8. Future Extensions

- Multi-modal integration (fMRI + sMRI)
- Temporal reasoning for longitudinal patient tracking
- Collaborative agent networks (multiple specialized agents)
- Active learning for model improvement
- Federated learning for privacy-preserving multi-site deployment

---

## Conclusion

The CDDA Framework represents a paradigm shift from passive ML prediction systems to active, reasoning-driven diagnostic support. By formalizing the five-layer architecture, core tool APIs, and autonomous decision logic, this specification provides a solid foundation for both implementation and thesis documentation.

**Next Steps:** Proceed to Phase 1 implementation - building the Tool Kit foundation with formalized API contracts.

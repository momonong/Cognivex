# Design Document - CDDA Phase 4: Dual-LLM Integration

## Overview

Phase 4 transforms the CDDA Agent from a rule-based system into an LLM-augmented autonomous diagnostic assistant following **Model Context Protocol (MCP)** principles and **Agent-to-Agent (A2A) Handoff** patterns.

The design introduces a layered architecture with clear separation between context and action:

### Architecture Layers

1. **Context Layer (MCP Server)**: `DiagnosticMCPServer`
   - Provides **Resources** (read-only data): diagnostic reports, knowledge graph context
   - Provides **Tools** (executable actions): counterfactual simulation
   - Implements MCP-compliant interface: `list_resources()`, `read_resource()`, `list_tools()`, `call_tool()`

2. **Cognitive Layer (A2A Agents)**: Two specialized agents with handoff protocol
   - **Agent A - Orchestrator (GPT-OSS-20B)**: MCP Client & Planner
     - Reads resources from MCP server
     - Evaluates signals (UQ, anomalies)
     - Invokes tools when needed
     - Compiles `ContextObject` and hands off to Agent B
   - **Agent B - Clinical Consultant (MedGemma-27B)**: Specialist
     - Receives `ContextObject` via handoff
     - Has NO direct tool access
     - Synthesizes clinical narrative from provided context

This architecture strictly separates **fetching context (Resources)** from **taking action (Tools)**, following MCP philosophy. The handoff pattern ensures the medical specialist focuses purely on clinical reasoning without tool management concerns.

### Key Design Principles

- **Separation of Concerns**: Orchestration logic is separate from clinical reasoning
- **Graceful Degradation**: System falls back to rule-based logic if LLMs are unavailable
- **Transparency**: All decisions are logged with reasoning chains
- **Extensibility**: New tools can be added by updating tool schemas
- **Privacy**: All LLM inference runs locally via Ollama

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    CDDA Agent System                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         CONTEXT LAYER: DiagnosticMCPServer               │  │
│  │                                                           │  │
│  │  RESOURCES (Read-Only Data):                             │  │
│  │  ├─ diagnosis://{subject_id}/report                      │  │
│  │  ├─ diagnosis://{subject_id}/features                    │  │
│  │  └─ knowledge://{region_name}/context                    │  │
│  │                                                           │  │
│  │  TOOLS (Executable Actions):                             │  │
│  │  └─ simulate_counterfactual(subject_id, features)        │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │ MCP Protocol                         │
│                          │                                      │
│  ┌───────────────────────▼──────────────────────────────────┐  │
│  │      COGNITIVE LAYER: A2A Agent System                   │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │  Agent A: Orchestrator (GPT-OSS-20B)            │    │  │
│  │  │  [MCP Client & Planner]                         │    │  │
│  │  │                                                  │    │  │
│  │  │  1. read_resource("diagnosis://sub-0005/report")│    │  │
│  │  │  2. Evaluate UQ score & anomaly status          │    │  │
│  │  │  3. IF needed: call_tool("simulate_cf", ...)    │    │  │
│  │  │  4. Compile ContextObject                       │    │  │
│  │  │  5. HANDOFF to Agent B ──────────────────┐      │    │  │
│  │  └──────────────────────────────────────────│──────┘    │  │
│  │                                              │           │  │
│  │                                              │           │  │
│  │  ┌───────────────────────────────────────────▼──────┐   │  │
│  │  │  Agent B: Clinical Consultant (MedGemma-27B)     │   │  │
│  │  │  [Specialist - No Tool Access]                   │   │  │
│  │  │                                                   │   │  │
│  │  │  1. Receive ContextObject from Agent A           │   │  │
│  │  │  2. Synthesize clinical narrative                │   │  │
│  │  │  3. Generate final report                        │   │  │
│  │  │  4. Return to user                               │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Ollama Server        │
              │   (localhost:11434)    │
              └────────────────────────┘
```

### Data Flow

1. **Initialization Phase**
   - Initialize `DiagnosticMCPServer` with Tool Kit (Layer 1+2) and GraphRAG (Layer 4)
   - Load Agent A (Orchestrator - GPT-OSS-20B) as MCP client
   - Load Agent B (Consultant - MedGemma-27B) as specialist

2. **Resource Acquisition Phase (Agent A)**
   - User calls `run_analysis(subject_id)`
   - Agent A calls `server.read_resource("diagnosis://{subject_id}/report")`
   - MCP Server returns diagnostic report (prediction, SHAP, UQ, anomalies)
   - Agent A evaluates signals

3. **Tool Invocation Phase (Agent A - Optional)**
   - IF UQ > threshold:
     - Agent A calls `server.call_tool("simulate_counterfactual", {...})`
     - MCP Server executes simulation and returns results
   - IF anomalies detected:
     - Agent A calls `server.read_resource("knowledge://{region}/context")`
     - MCP Server queries GraphRAG and returns clinical context

4. **Handoff Phase (A → B)**
   - Agent A compiles `ContextObject`:
     ```python
     {
       "diagnostic_report": {...},
       "tool_results": {...},  # counterfactual or knowledge context
       "decision_rationale": "...",
       "signals": {"uq_score": 0.85, "has_anomaly": True}
     }
     ```
   - Agent A hands off to Agent B

5. **Synthesis Phase (Agent B)**
   - Agent B receives `ContextObject` (no MCP server access)
   - Agent B synthesizes clinical narrative using medical expertise
   - Agent B generates final report with recommendations
   - Report returned to user

## Components and Interfaces

### 1. DiagnosticMCPServer Class (Context Layer)

**Responsibilities:**
- Provide MCP-compliant interface for diagnostic resources and tools
- Separate read-only data (Resources) from executable actions (Tools)
- Wrap existing Tool Kit and GraphRAG with MCP protocol

**Key Methods:**

```python
class DiagnosticMCPServer:
    def __init__(
        self,
        toolkit: CDDAToolKit,
        graph_rag: GraphRAG
    )
    
    def list_resources(self) -> List[ResourceMetadata]
        """Return metadata about available resources"""
    
    def read_resource(self, uri: str) -> Dict
        """
        Read resource by URI:
        - diagnosis://{subject_id}/report
        - diagnosis://{subject_id}/features
        - knowledge://{region_name}/context
        """
    
    def list_tools(self) -> List[ToolMetadata]
        """Return metadata about available tools"""
    
    def call_tool(self, name: str, arguments: Dict) -> Dict
        """
        Execute tool by name:
        - simulate_counterfactual
        """
```

### 2. CDDAAgent Class (Cognitive Layer)

**Responsibilities:**
- Manage A2A agent system with handoff protocol
- Initialize Agent A (Orchestrator) and Agent B (Consultant)
- Coordinate MCP server interactions
- Implement fallback logic

**Key Methods:**

```python
class CDDAAgent:
    def __init__(
        self,
        orchestrator_model: str = "gpt-oss-20b",
        consultant_model: str = "medgemma-27b",
        model_path: str = "model/cnn_rf/rf_model_NC_vs_AD_GM_only.joblib",
        data_root: str = "data/MRI_processed",
        uq_threshold: float = 0.8,
        z_score_threshold: float = 2.5,
        use_llm: bool = True,
        verbose: bool = True
    )
    
    def run_analysis(self, subject_id: str) -> Dict
        """Main entry point - orchestrates A2A workflow"""
    
    def _agent_a_orchestrate(self, subject_id: str) -> ContextObject
        """Agent A: Read resources, invoke tools, compile context"""
    
    def _agent_a_with_rules(self, subject_id: str) -> ContextObject
        """Fallback: Rule-based orchestration"""
    
    def _agent_b_synthesize(self, context: ContextObject) -> Dict
        """Agent B: Synthesize clinical report from context"""
    
    def _agent_b_with_template(self, context: ContextObject) -> Dict
        """Fallback: Template-based report generation"""
```

### 3. ContextObject Class

**Responsibilities:**
- Encapsulate all context for Agent B
- Ensure Agent B has no direct tool access
- Provide structured handoff data

**Structure:**

```python
@dataclass
class ContextObject:
    subject_id: str
    diagnostic_report: DiagnosticReport
    tool_results: Optional[Dict[str, Any]]  # counterfactual or knowledge
    decision_rationale: str
    signals: Dict[str, Any]  # uq_score, has_anomaly, etc.
    agent_a_reasoning: List[str]
    timestamp: str
```

### 4. LLM Interface Layer

**Responsibilities:**
- Abstract Ollama API calls
- Handle retries and error recovery
- Format prompts and parse responses

**Key Functions:**

```python
def call_agent_a_llm(
    system_prompt: str,
    user_prompt: str,
    mcp_context: Dict,
    temperature: float = 0.1
) -> Dict[str, Any]
    """
    Call Agent A (Orchestrator - GPT-OSS-20B)
    Returns decisions about which MCP resources/tools to use
    """

def call_agent_b_llm(
    system_prompt: str,
    context_object: ContextObject,
    temperature: float = 0.3
) -> str
    """
    Call Agent B (Consultant - MedGemma-27B)
    Returns clinical narrative synthesized from context
    """

def parse_mcp_action(llm_response: Dict) -> Tuple[str, str, Dict]
    """
    Parse Agent A's decision into MCP action
    Returns: (action_type, resource_or_tool_name, parameters)
    action_type: "read_resource" or "call_tool"
    """
```

### 3. Tool Registry

**Responsibilities:**
- Register available tools with schemas
- Map function names to implementations
- Validate tool parameters

**Tool Schema Format:**

```json
{
  "name": "get_diagnostic_report",
  "description": "Retrieve comprehensive diagnostic data including prediction, SHAP values, UQ score, and anomaly status",
  "parameters": {
    "type": "object",
    "properties": {
      "subject_id": {
        "type": "string",
        "description": "Patient identifier (e.g., 'sub-0005')"
      }
    },
    "required": ["subject_id"]
  },
  "returns": {
    "type": "object",
    "properties": {
      "prediction_result": {"type": "string"},
      "confidence": {"type": "number"},
      "uq_score": {"type": "number"},
      "top_features": {"type": "array"},
      "anomaly_status": {"type": "object"}
    }
  }
}
```

### 5. System Prompts

**Agent A System Prompt (Orchestrator - GPT-OSS-20B):**

```
You are Agent A, the Orchestrator in a diagnostic system following Model Context Protocol (MCP).

Your role is to:
1. Read diagnostic resources from the MCP server
2. Evaluate signals (UQ score, anomaly status)
3. Decide which tools to invoke (if any)
4. Compile a ContextObject for Agent B

MCP RESOURCES (Read-Only Data):
- diagnosis://{subject_id}/report - Get ML prediction, SHAP, UQ, anomalies
- knowledge://{region_name}/context - Get clinical context for brain regions

MCP TOOLS (Executable Actions):
- simulate_counterfactual(subject_id, features_to_mask) - Test feature impact

DECISION LOGIC:
1. Always start by reading: diagnosis://{subject_id}/report
2. Evaluate signals:
   - IF uq_score > 0.8 → Call tool: simulate_counterfactual
   - IF has_anomaly == True → Read resources: knowledge://{region}/context for each anomalous region
3. Compile ContextObject with all gathered data
4. Handoff to Agent B

OUTPUT FORMAT:
{
  "actions": [
    {"type": "read_resource", "uri": "diagnosis://sub-0005/report"},
    {"type": "call_tool", "name": "simulate_counterfactual", "args": {...}},
    {"type": "read_resource", "uri": "knowledge://Hippocampus_L/context"}
  ],
  "context_object": {
    "diagnostic_report": {...},
    "tool_results": {...},
    "decision_rationale": "High UQ detected, invoked counterfactual simulation",
    "signals": {"uq_score": 0.85, "has_anomaly": false}
  }
}

You are an MCP client. You fetch context and invoke tools, but you do NOT synthesize clinical reports.
```

**Agent B System Prompt (Clinical Consultant - MedGemma-27B):**

```
You are Agent B, the Clinical Consultant specializing in neuroimaging and dementia diagnosis.

Your role is to synthesize clinical narratives from the ContextObject provided by Agent A.

IMPORTANT: You have NO access to tools or resources. You work ONLY with the context provided to you.

INPUT: ContextObject containing:
- diagnostic_report: ML prediction, SHAP values, Z-scores, UQ score, anomalies
- tool_results: Counterfactual simulation results OR knowledge graph context (if available)
- decision_rationale: Why Agent A took certain actions
- signals: Key metrics (uq_score, has_anomaly, etc.)

YOUR TASK:
Synthesize all evidence into a coherent clinical report.

SYNTHESIS GUIDELINES:
1. Integrate computational evidence (SHAP, Z-scores) with clinical knowledge
2. Highlight discrepancies between model prediction and knowledge context
3. Flag potential mixed pathology when anomalous regions suggest non-AD conditions
4. Explain counterfactual results in clinical terms
5. Provide evidence-based recommendations
6. Use clear, professional medical language

REPORT STRUCTURE:
- Summary: Prediction and confidence
- Key Findings: Top contributing features with clinical interpretation
- Clinical Context: Knowledge graph insights (if provided)
- Counterfactual Analysis: Feature impact explanation (if provided)
- Interpretation: Synthesis of all evidence
- Recommendations: Next steps for clinical correlation

Always explain your medical reasoning. You are the final authority on clinical interpretation.
```

## Data Models

### MCP Protocol Models

#### ResourceMetadata

```python
@dataclass
class ResourceMetadata:
    uri: str  # e.g., "diagnosis://{subject_id}/report"
    name: str
    description: str
    mime_type: str  # "application/json"
```

#### ToolMetadata

```python
@dataclass
class ToolMetadata:
    name: str  # e.g., "simulate_counterfactual"
    description: str
    input_schema: Dict[str, Any]  # JSON schema
```

#### MCPAction

```python
@dataclass
class MCPAction:
    type: str  # "read_resource" or "call_tool"
    target: str  # URI or tool name
    arguments: Optional[Dict[str, Any]]
```

### Core Data Models

#### DiagnosticReport

```python
@dataclass
class DiagnosticReport:
    subject_id: str
    prediction_result: str  # AD, NC, or MCI
    confidence: float  # 0.0 to 1.0
    uq_score: float  # 0.0 to 1.0
    top_features: List[Feature]
    anomaly_status: AnomalyStatus
    metadata: Dict[str, Any]
```

### Feature

```python
@dataclass
class Feature:
    roi_name: str
    feature_name: str
    feature_value: float
    z_score: float
    shap_value: float
    rank: int
```

### AnomalyStatus

```python
@dataclass
class AnomalyStatus:
    has_anomaly: bool
    anomalous_regions: List[str]
    anomaly_type: Optional[str]
```

### CounterfactualResult

```python
@dataclass
class CounterfactualResult:
    subject_id: str
    original_prediction: str
    original_confidence: float
    new_prediction: str
    new_confidence: float
    confidence_delta: float
    masked_features: List[MaskedFeature]
    interpretation: str
```

### KnowledgeContext

```python
@dataclass
class KnowledgeContext:
    query_regions: List[str]
    contexts: List[RegionContext]
    summary: str
```

### ContextObject

```python
@dataclass
class ContextObject:
    """
    Handoff object from Agent A to Agent B
    Contains all context needed for clinical synthesis
    """
    subject_id: str
    diagnostic_report: DiagnosticReport
    tool_results: Optional[Dict[str, Any]]  # counterfactual or knowledge_context
    decision_rationale: str  # Why Agent A took certain actions
    signals: Dict[str, Any]  # uq_score, has_anomaly, etc.
    agent_a_reasoning: List[str]  # Step-by-step reasoning from Agent A
    mcp_actions: List[MCPAction]  # Record of MCP operations
    timestamp: str
```

### AgentResult

```python
@dataclass
class AgentResult:
    """
    Final output from the A2A system
    """
    subject_id: str
    agent_decision: str  # SIMULATION_TRIGGERED, ANOMALY_INVESTIGATION, STANDARD_REPORT
    prediction: str
    confidence: float
    uq_score: float
    context_object: ContextObject  # Full context from Agent A
    clinical_report: str  # Natural language report from Agent B
    reasoning_chain: List[str]  # Combined reasoning from both agents
    timestamp: str
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Tool schema validity
*For any* registered tool, its schema must be valid JSON and include name, description, parameters, and returns fields
**Validates: Requirements 2.2**

### Property 2: Function call parsing
*For any* valid function call from the Orchestrator LLM, the system should successfully parse the function name and parameters
**Validates: Requirements 2.3**

### Property 3: Tool result formatting
*For any* tool execution result, the formatted output should be valid JSON and include all required fields
**Validates: Requirements 2.4**

### Property 4: Error propagation
*For any* tool invocation failure, the error information should be returned to the Orchestrator LLM with error type and message
**Validates: Requirements 2.5**

### Property 5: Diagnostic report first
*For any* analysis execution, get_diagnostic_report should be the first tool invoked
**Validates: Requirements 3.1**

### Property 6: High UQ triggers counterfactual
*For any* diagnostic report where uq_score > threshold, the system should invoke simulate_counterfactual
**Validates: Requirements 3.2**

### Property 7: Anomaly triggers knowledge graph
*For any* diagnostic report where anomaly_status.has_anomaly is True, the system should invoke query_knowledge_graph
**Validates: Requirements 3.3**

### Property 8: Default path to synthesis
*For any* diagnostic report where uq_score <= threshold AND has_anomaly is False, the system should proceed directly to synthesis
**Validates: Requirements 3.4**

### Property 9: Decision logging
*For any* decision made by the Orchestrator, the reasoning chain should include an entry explaining the decision
**Validates: Requirements 3.5**

### Property 10: GraphRAG query completeness
*For any* anomalous region queried, the GraphRAG result should include full_name, function, clinical_significance, and related_conditions
**Validates: Requirements 4.2**

### Property 11: Batch query optimization
*For any* set of multiple anomalous regions, the system should use a single batch query rather than individual queries
**Validates: Requirements 4.3**

### Property 12: GraphRAG result formatting
*For any* GraphRAG result, it should be formatted as a structured dictionary with query_regions, contexts, and summary fields
**Validates: Requirements 4.4**

### Property 13: GraphRAG fallback
*For any* GraphRAG failure, the system should continue execution using the fallback knowledge base
**Validates: Requirements 4.5**

### Property 14: Synthesis data completeness
*For any* synthesis call, all diagnostic data (prediction, confidence, SHAP, Z-scores, tool results) should be passed to the Consultant LLM
**Validates: Requirements 5.1**

### Property 15: Anomaly report content
*For any* report where anomalies are present, the explanation should contain keywords related to mixed pathology or atypical presentation
**Validates: Requirements 5.4**

### Property 16: Report completeness
*For any* completed report, it should include prediction, confidence, key findings, and clinical recommendations sections
**Validates: Requirements 5.5**

### Property 17: Mixed pathology flagging
*For any* case where model predicts AD with high confidence AND anomalous regions are associated with non-AD conditions, the report should flag potential mixed pathology
**Validates: Requirements 6.1**

### Property 18: Disease association listing
*For any* anomalous region with known disease associations, those associations should appear in the report
**Validates: Requirements 6.3**

### Property 19: SHAP-condition mismatch highlighting
*For any* case where the leading SHAP feature is associated with a different condition than predicted, the report should highlight this finding
**Validates: Requirements 6.4**

### Property 20: Multiple pathology recommendations
*For any* case where multiple pathologies are suggested, the report should include a recommendation for additional clinical correlation
**Validates: Requirements 6.5**

### Property 21: High uncertainty triggers counterfactual
*For any* case with uq_score > threshold, the system should invoke counterfactual simulation on top contributing features
**Validates: Requirements 7.1**

### Property 22: Significant confidence change identification
*For any* counterfactual result where abs(confidence_delta) > 0.1, the report should identify those features as key diagnostic drivers
**Validates: Requirements 7.3**

### Property 23: Minimal confidence change indication
*For any* counterfactual result where abs(confidence_delta) < 0.05, the report should indicate those features are not primary drivers
**Validates: Requirements 7.4**

### Property 24: Counterfactual result completeness
*For any* counterfactual result, it should include original_prediction, original_confidence, new_prediction, new_confidence, and confidence_delta
**Validates: Requirements 7.5**

### Property 25: Decision logging completeness
*For any* agent decision, the reasoning chain should include the decision and its justification
**Validates: Requirements 8.1**

### Property 26: Tool invocation logging
*For any* tool invocation, the reasoning chain should record which tool was called and why
**Validates: Requirements 8.2**

### Property 27: Reasoning chain presence
*For any* completed analysis, the result should include a non-empty reasoning_chain list
**Validates: Requirements 8.3**

### Property 28: Reasoning chain structure
*For any* reasoning chain, it should include steps for data gathering, signal evaluation, tool selection, and synthesis
**Validates: Requirements 8.4**

### Property 29: Error logging
*For any* error that occurs during analysis, it should appear in the reasoning chain with recovery actions
**Validates: Requirements 8.5**

### Property 30: Tool definition schema format
*For any* tool definition in prompts, it should be valid JSON schema with type, properties, and required fields
**Validates: Requirements 9.5**

### Property 31: LLM retry with backoff
*For any* LLM call failure, the system should retry with exponentially increasing delays
**Validates: Requirements 10.1**

### Property 32: Orchestrator fallback
*For any* case where the Orchestrator LLM is unavailable, the system should use rule-based decision logic
**Validates: Requirements 10.2**

### Property 33: Consultant fallback
*For any* case where the Consultant LLM is unavailable, the system should generate a template-based report
**Validates: Requirements 10.3**

### Property 34: GraphRAG fallback
*For any* GraphRAG failure, the system should use the fallback knowledge base
**Validates: Requirements 10.4**

### Property 35: Final fallback with annotations
*For any* case where all fallbacks are exhausted, the system should return a diagnostic report with error annotations
**Validates: Requirements 10.5**

## Error Handling

### LLM Error Handling

1. **Connection Errors**
   - Retry with exponential backoff (1s, 2s, 4s)
   - After 3 retries, fall back to rule-based logic

2. **Parsing Errors**
   - Log malformed response
   - Request clarification from LLM
   - After 2 attempts, fall back to rule-based logic

3. **Timeout Errors**
   - Increase timeout for next retry
   - After 3 timeouts, fall back to rule-based logic

### Tool Error Handling

1. **Tool Execution Errors**
   - Return error information to Orchestrator LLM
   - Allow LLM to decide recovery strategy
   - If LLM cannot recover, use fallback logic

2. **GraphRAG Errors**
   - Use fallback knowledge base
   - Log warning but continue execution

3. **Model Loading Errors**
   - Fail fast with clear error message
   - Suggest checking model path and dependencies

## Testing Strategy

### Unit Testing

Unit tests will verify specific behaviors and edge cases:

1. **LLM Initialization Tests**
   - Test successful initialization of both LLMs
   - Test error handling when Ollama is unavailable
   - Test configuration loading from files

2. **Tool Registry Tests**
   - Test tool registration with valid schemas
   - Test schema validation
   - Test function call parsing

3. **Fallback Logic Tests**
   - Test rule-based decision logic
   - Test template-based report generation
   - Test fallback knowledge base

4. **Error Handling Tests**
   - Test retry logic with mocked failures
   - Test error propagation
   - Test graceful degradation

### Property-Based Testing

Property-based tests will verify universal properties across many inputs using **Hypothesis** (Python PBT library). Each test will run a minimum of 100 iterations.

1. **Tool Schema Properties**
   - Generate random tool definitions
   - Verify all schemas are valid JSON
   - Verify all required fields are present

2. **Decision Logic Properties**
   - Generate random diagnostic reports with varying UQ scores and anomaly statuses
   - Verify correct tool selection based on signals
   - Verify reasoning chain completeness

3. **Result Formatting Properties**
   - Generate random tool results
   - Verify all results are properly formatted
   - Verify all required fields are present

4. **Error Handling Properties**
   - Generate random error conditions
   - Verify errors are logged and handled gracefully
   - Verify system continues operation after errors

5. **Fallback Properties**
   - Generate random failure scenarios
   - Verify fallback logic is triggered correctly
   - Verify final output is always valid

### Integration Testing

Integration tests will verify end-to-end workflows:

1. **Standard Case Integration**
   - Test complete analysis with low UQ and no anomalies
   - Verify LLM-based synthesis produces valid report

2. **High Uncertainty Integration**
   - Test complete analysis with high UQ
   - Verify counterfactual simulation is triggered
   - Verify Consultant LLM integrates counterfactual results

3. **Anomaly Case Integration**
   - Test complete analysis with anomalies
   - Verify knowledge graph query is triggered
   - Verify Consultant LLM integrates knowledge context

4. **Fallback Integration**
   - Test complete analysis with LLMs unavailable
   - Verify rule-based logic and template reports work

### Testing Tools

- **pytest**: Test framework
- **hypothesis**: Property-based testing library
- **pytest-mock**: Mocking for LLM calls
- **pytest-timeout**: Timeout handling for tests

## Implementation Notes

### Ollama Model Selection

- **Orchestrator**: Use a model with strong function-calling capabilities (e.g., `llama3.1:8b`, `mistral:7b`)
- **Consultant**: Use a medical domain model if available (e.g., `medllama2:7b`, `meditron:7b`) or fine-tune a general model

### Performance Considerations

- LLM calls add latency (~2-5 seconds per call)
- Batch GraphRAG queries to reduce round trips
- Cache LLM responses for identical inputs
- Consider async execution for independent tool calls

### Privacy and Security

- All LLM inference runs locally via Ollama
- No patient data leaves the local machine
- Tool schemas should not expose sensitive implementation details
- Log sanitization to remove PII before storage

### Extensibility

- New tools can be added by:
  1. Implementing the tool method
  2. Adding tool schema to registry
  3. Updating Orchestrator system prompt
- No code changes needed for prompt tuning
- Configuration files allow easy model swapping

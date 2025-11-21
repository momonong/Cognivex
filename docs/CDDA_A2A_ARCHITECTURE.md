# CDDA A2A Architecture

## Overview

The CDDA Agent now implements an Agent-to-Agent (A2A) pattern with dual-LLM architecture, following Model Context Protocol (MCP) principles.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CDDA Agent (A2A Pattern)                      │
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
```

## Data Flow

### Phase 1: Agent A Orchestration

```
User Request
    ↓
Agent A.orchestrate(subject_id)
    ↓
MCP Server.read_resource("diagnosis://sub-0005/report")
    ↓
Evaluate Signals (UQ, Anomalies)
    ↓
IF UQ > threshold:
    MCP Server.call_tool("simulate_counterfactual", ...)
    ↓
IF Anomalies detected:
    MCP Server.read_resource("knowledge://region/context")
    ↓
Compile ContextObject
    ├─ diagnostic_report
    ├─ tool_results (counterfactual or knowledge)
    ├─ decision_rationale
    ├─ signals
    ├─ agent_a_reasoning (with timestamps)
    └─ mcp_actions (with timestamps)
```

### Phase 2: Handoff

```
Agent A → Agent B
    ↓
ContextObject passed
    ↓
Agent B has NO direct tool access
Agent B works ONLY with provided context
```

### Phase 3: Agent B Synthesis

```
Agent B.synthesize(context_object)
    ↓
Format context for LLM (or use template)
    ↓
Generate clinical narrative
    ├─ Integrate SHAP values
    ├─ Interpret Z-scores
    ├─ Explain counterfactual results
    ├─ Flag anomalies and mixed pathology
    └─ Provide clinical recommendations
    ↓
Return clinical_report + agent_b_reasoning
```

### Phase 4: Reasoning Chain Aggregation

```
Combine reasoning chains:
    ├─ Agent A orchestration steps
    ├─ MCP actions (with timestamps)
    ├─ Handoff documentation
    └─ Agent B synthesis steps
    ↓
Build AgentResult
    ├─ subject_id
    ├─ agent_decision
    ├─ prediction
    ├─ confidence
    ├─ uq_score
    ├─ context_object
    ├─ clinical_report
    ├─ reasoning_chain (combined)
    └─ timestamp
    ↓
Save reasoning log to JSON (optional)
```

## Key Design Principles

### 1. Separation of Concerns

- **Agent A**: Orchestration, tool management, data gathering
- **Agent B**: Clinical reasoning, narrative synthesis
- **MCP Server**: Context and action provider

### 2. No Direct Tool Access for Agent B

Agent B receives a **ContextObject** containing:
- Diagnostic report
- Tool results (if any)
- Decision rationale
- Signals (UQ, anomalies)
- Agent A's reasoning

This ensures Agent B focuses purely on clinical interpretation without tool management concerns.

### 3. Complete Transparency

Every step is logged with timestamps:
- Agent A decisions
- MCP actions (read_resource, call_tool)
- Handoff details
- Agent B synthesis steps

### 4. Graceful Degradation

Multiple fallback levels:
1. LLM-based orchestration (Agent A with GPT-OSS-20B)
2. Rule-based orchestration (if LLM unavailable)
3. LLM-based synthesis (Agent B with MedGemma-27B)
4. Template-based synthesis (if LLM unavailable)

## Example Reasoning Chain

```
================================================================================
AGENT A - ORCHESTRATION
================================================================================
[2025-11-20T20:46:05.724902] Starting orchestration for sub-0015
[2025-11-20T20:46:10.809848] Read diagnostic report for sub-0015
[2025-11-20T20:46:10.809848] Evaluated signals: UQ=0.667, Anomaly=False
[2025-11-20T20:46:10.809848] Standard case: low uncertainty, no anomalies

--------------------------------------------------------------------------------
MCP ACTIONS
--------------------------------------------------------------------------------
[2025-11-20T20:46:05.724902] read_resource: diagnosis://sub-0015/report → success

--------------------------------------------------------------------------------
HANDOFF: Agent A → Agent B
--------------------------------------------------------------------------------
Decision Rationale: Standard case: low uncertainty, no anomalies
Context Object validated: True

================================================================================
AGENT B - CLINICAL SYNTHESIS
================================================================================
[2025-11-20T20:46:10.810848] Received ContextObject for sub-0015
[2025-11-20T20:46:10.810848] Prediction: NC
[2025-11-20T20:46:10.810848] Confidence: 76.1%
[2025-11-20T20:46:10.810848] UQ Score: 0.667
[2025-11-20T20:46:10.810848] Template-based synthesis completed
```

## Benefits

### For Development
- Clear separation of concerns
- Easy to test each agent independently
- Modular architecture for future extensions

### For Research
- Complete audit trail for paper evidence
- Structured reasoning logs in JSON format
- Timestamps for all decisions and actions

### For Clinical Use
- Transparent decision-making process
- Explainable AI with reasoning chains
- Graceful degradation ensures reliability

## Usage Example

```python
from app.agents.cdda_agent import CDDAAgent

# Initialize with A2A architecture
agent = CDDAAgent(
    orchestrator_model="gpt-oss-20b",
    consultant_model="medgemma-27b",
    use_llm=True,  # Enable LLM-based agents
    verbose=True
)

# Run analysis
result = agent.run_analysis('sub-0005')

# Access results
print(f"Prediction: {result.prediction}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Decision: {result.agent_decision}")

# Print clinical report
print(result.clinical_report)

# Print reasoning chain
for step in result.reasoning_chain:
    print(step)

# Save reasoning log for paper
agent.save_reasoning_log(result, "output/reasoning_log.json")
```

## Next Steps

1. **Task 6**: System prompt configuration
2. **Task 7**: Error handling and fallbacks
3. **Task 8**: Integration testing
4. **Task 9**: Documentation and demos

## References

- Design Document: `.kiro/specs/cdda-phase4-dual-llm/design.md`
- Requirements: `.kiro/specs/cdda-phase4-dual-llm/requirements.md`
- Tasks: `.kiro/specs/cdda-phase4-dual-llm/tasks.md`
- Implementation: `app/agents/cdda_agent.py`

# CDDA Phase 4: Dual-LLM Integration - Planning Complete

## Overview

Phase 4 specification is complete! This phase transforms the CDDA Agent into an MCP-compliant, A2A-based system with dual-LLM architecture following industry-standard patterns.

## Architecture Summary

### Model Context Protocol (MCP) Compliance

**Context Layer: DiagnosticMCPServer**
- Separates Resources (read-only data) from Tools (executable actions)
- Resources:
  - `diagnosis://{subject_id}/report` - Diagnostic data with SHAP, UQ, anomalies
  - `knowledge://{region_name}/context` - Clinical context from knowledge graph
- Tools:
  - `simulate_counterfactual` - What-if analysis for feature impact

### Agent-to-Agent (A2A) Handoff Pattern

**Agent A: Orchestrator (GPT-OSS-20B)**
- Role: MCP Client & Planner
- Responsibilities:
  - Read resources from MCP server
  - Evaluate signals (UQ score, anomaly status)
  - Invoke tools when needed
  - Compile ContextObject for Agent B
- Has: Full MCP server access

**Agent B: Clinical Consultant (MedGemma-27B)**
- Role: Medical Specialist
- Responsibilities:
  - Receive ContextObject from Agent A
  - Synthesize clinical narrative
  - Generate final report with medical reasoning
- Has: NO tool access (pure specialist)

### Handoff Protocol

```
Agent A (Orchestrator)
  ↓ read_resource("diagnosis://sub-0005/report")
  ↓ evaluate signals
  ↓ call_tool("simulate_counterfactual") [if needed]
  ↓ compile ContextObject
  ↓ HANDOFF
Agent B (Consultant)
  ↓ receive ContextObject
  ↓ synthesize clinical report
  ↓ return to user
```

## Key Design Decisions

### 1. Medical Expert-Assistant Strategy
- **MedGemma-27B** is the brain (medical reasoning, clinical synthesis)
- **GPT-OSS-20B** is the hands (MCP client, tool orchestration)
- Leverages each model's strengths

### 2. MCP Compliance
- Strict separation of Resources vs. Tools
- URI-based resource addressing
- JSON schema for tool definitions
- Follows industry-standard protocol

### 3. A2A Handoff
- Agent B is a pure specialist with no tool management
- ContextObject ensures complete context transfer
- Clear separation of concerns

### 4. Graceful Degradation
- Agent A fallback: Rule-based orchestration
- Agent B fallback: Template-based reports
- GraphRAG fallback: Mock knowledge base
- System never fails completely

## Implementation Phases

### Phase 1-2: Foundation (MCP Server + Data Models)
- DiagnosticMCPServer with MCP-compliant interface
- ContextObject and MCP data models
- Resource URI routing and tool invocation

### Phase 3-4: Agents (Orchestrator + Consultant)
- Agent A with LLM integration and MCP client logic
- Agent B with medical domain synthesis
- Structured logging for paper evidence (HIGH ROI)

### Phase 5: Integration (A2A Handoff)
- CDDAAgent coordination layer
- Handoff protocol implementation
- Reasoning chain aggregation (HIGH ROI)

### Phase 6-7: Configuration + Error Handling
- System prompts for both agents
- Tool schemas in JSON
- Retry logic with exponential backoff (HIGH ROI)
- Comprehensive fallback mechanisms

### Phase 8-10: Testing + Documentation
- Integration tests for MCP compliance
- A2A handoff tests
- Fallback scenario tests
- Comprehensive documentation

## Priority Guidelines

### HIGH ROI (Must Do)
✅ **Structured Logging**
- Essential for paper evidence
- Demonstrates agent reasoning traces
- Enables observability

✅ **Error Recovery**
- Essential for demo stability
- GPT-OSS may output malformed JSON
- Retry logic prevents crashes

✅ **Core Implementation**
- MCP server, A2A agents, handoff protocol
- Minimum viable system

### MEDIUM ROI (Optional)
- Property-based tests (marked with *)
- Can be added incrementally

### LOW ROI (Skip for MVP)
❌ **Parallel Tool Execution**
- Risk of VRAM OOM on single GPU
- Sequential execution is safer

❌ **Streaming Responses**
- Pure UI enhancement
- No academic value for paper

## Correctness Properties

35 properties defined covering:
- Tool schema validity and function call parsing
- Decision logic (UQ triggers, anomaly triggers)
- MCP resource completeness
- GraphRAG fallback behavior
- Synthesis data completeness
- Anomaly awareness and mixed pathology flagging
- Counterfactual interpretation
- Reasoning chain structure
- Error handling and fallback mechanisms

## Next Steps

1. **Start Implementation**: Begin with Phase 1 (MCP Server Foundation)
2. **Focus on High ROI**: Prioritize structured logging and error recovery
3. **Incremental Testing**: Add tests as you implement each phase
4. **Documentation**: Keep reasoning traces for paper evidence

## Files Created

- `.kiro/specs/cdda-phase4-dual-llm/requirements.md` - Complete requirements with EARS patterns
- `.kiro/specs/cdda-phase4-dual-llm/design.md` - Detailed design with MCP and A2A architecture
- `.kiro/specs/cdda-phase4-dual-llm/tasks.md` - Implementation plan with 10 phases

## Benefits of This Architecture

1. **MCP Compliance**: Industry-standard protocol for context and tools
2. **Clear Separation**: Resources vs. Tools, Planning vs. Synthesis
3. **Medical Expertise**: MedGemma drives all clinical decisions
4. **Fault Tolerance**: Comprehensive fallback mechanisms
5. **Observability**: Structured logging for paper evidence
6. **Extensibility**: Easy to add new resources or tools
7. **Privacy**: All LLM inference runs locally via Ollama

## Ready to Implement!

The specification is complete and ready for implementation. You can start executing tasks by opening `tasks.md` and clicking "Start task" next to task items.

**Recommended Starting Point**: Task 1 - Implement DiagnosticMCPServer class

---

*Planning completed: 2024*
*Architecture: MCP + A2A with Dual-LLM*
*Status: Ready for Implementation*

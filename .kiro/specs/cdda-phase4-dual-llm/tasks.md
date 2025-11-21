# Implementation Plan - CDDA Phase 4: Dual-LLM Integration

## Overview

This implementation plan transforms the CDDA Agent into an MCP-compliant, A2A-based system with dual-LLM architecture. Tasks are organized to build incrementally: first the MCP server layer, then the A2A agent layer, and finally integration and testing.

### Task Priority Guidelines

**HIGH ROI (Must Do):**
- ✅ Structured Logging: Essential for paper evidence and debugging
- ✅ Error Recovery: Essential for demo stability (GPT-OSS may output malformed JSON)
- ✅ Core Implementation: MCP server, A2A agents, handoff protocol

**MEDIUM ROI (Optional - Marked with *):**
- Property-based tests: Good for robustness but can be added later
- Integration tests: Important but can be done incrementally

**LOW ROI (Skip for MVP):**
- ❌ Parallel Tool Execution: Risk of VRAM OOM on single GPU, sequential is safer
- ❌ Streaming Responses: Pure UI enhancement, no academic value for paper

**Note:** Tasks marked with `*` are optional and can be skipped for faster MVP development. High ROI tasks are explicitly marked and should be prioritized.

---

## Phase 1: MCP Server Foundation

- [x] 1. Implement DiagnosticMCPServer class





  - Create `app/core/mcp_server.py` with MCP-compliant interface
  - Implement `list_resources()` method returning resource metadata
  - Implement `read_resource(uri)` method with URI parsing
  - Implement `list_tools()` method returning tool metadata
  - Implement `call_tool(name, arguments)` method with tool execution
  - _Requirements: 2.1, 4.1, 4.2_

- [ ]* 1.1 Write property test for MCP resource URIs
  - **Property 1: Tool schema validity**
  - **Validates: Requirements 2.2**

- [x] 1.2 Implement resource URI routing

  - Add URI parser for `diagnosis://{subject_id}/report` pattern
  - Add URI parser for `knowledge://{region_name}/context` pattern
  - Route to appropriate backend (CDDAToolKit or GraphRAG)
  - Handle invalid URIs with clear error messages
  - _Requirements: 4.1, 4.2_

- [ ]* 1.3 Write property test for URI parsing
  - **Property 2: Function call parsing**
  - **Validates: Requirements 2.3**


- [ ] 1.4 Implement tool invocation layer
  - Map tool name "simulate_counterfactual" to CDDAToolKit method
  - Validate tool arguments against schema
  - Execute tool and return structured results
  - Handle tool execution errors gracefully
  - _Requirements: 2.4, 2.5_

- [ ]* 1.5 Write property test for tool execution
  - **Property 3: Tool result formatting**
  - **Property 4: Error propagation**
  - **Validates: Requirements 2.4, 2.5**

---

## Phase 2: Data Models and Context Objects

- [x] 2. Define MCP and A2A data models





  - Create `app/core/models/mcp_models.py` with ResourceMetadata, ToolMetadata, MCPAction
  - Create `app/core/models/context_models.py` with ContextObject
  - Update AgentResult model to include ContextObject
  - Add JSON serialization methods for all models
  - _Requirements: 3.5, 8.1, 8.2_

- [ ]* 2.1 Write property test for data model serialization
  - **Property 24: Counterfactual result completeness**
  - **Validates: Requirements 7.5**

- [x] 2.2 Implement ContextObject builder



  - Create helper class to compile ContextObject from diagnostic data
  - Add validation to ensure all required fields are present
  - Add method to serialize ContextObject for Agent B
  - _Requirements: 5.1, 8.3_

- [ ]* 2.3 Write property test for ContextObject completeness
  - **Property 14: Synthesis data completeness**
  - **Validates: Requirements 5.1**

---

## Phase 3: Agent A (Orchestrator) Implementation

- [x] 3. Implement Agent A orchestration logic






  - Create `app/agents/agent_a_orchestrator.py` with AgentA class
  - Implement `orchestrate(subject_id)` method
  - Add MCP client logic to read resources and call tools
  - Implement decision logic based on UQ score and anomaly status
  - Compile ContextObject for handoff to Agent B
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ]* 3.1 Write property test for decision logic
  - **Property 5: Diagnostic report first**
  - **Property 6: High UQ triggers counterfactual**
  - **Property 7: Anomaly triggers knowledge graph**
  - **Property 8: Default path to synthesis**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [x] 3.2 Implement Agent A LLM integration

  - Add Ollama client for GPT-OSS-20B model
  - Implement system prompt loading from config
  - Add prompt formatting with MCP context
  - Parse LLM response into MCPAction list
  - _Requirements: 1.1, 1.2, 9.1_

- [ ]* 3.3 Write property test for LLM response parsing
  - **Property 2: Function call parsing**
  - **Validates: Requirements 2.3**

- [x] 3.4 Implement structured reasoning chain logging (HIGH ROI - Required for paper)

  - Add structured JSON logging for each MCP action (read_resource, call_tool)
  - Log decision rationale for each action with timestamps
  - Log all Agent A decisions with reasoning traces
  - Compile complete reasoning chain into ContextObject
  - Save logs to file for paper evidence generation
  - _Requirements: 3.5, 8.1, 8.2_
  - _Note: Essential for demonstrating agent reasoning in paper_

- [ ]* 3.5 Write property test for reasoning chain
  - **Property 9: Decision logging**
  - **Property 25: Decision logging completeness**
  - **Property 26: Tool invocation logging**
  - **Validates: Requirements 3.5, 8.1, 8.2**

---

## Phase 4: Agent B (Consultant) Implementation

- [x] 4. Implement Agent B synthesis logic





  - Create `app/agents/agent_b_consultant.py` with AgentB class
  - Implement `synthesize(context_object)` method
  - Ensure Agent B has NO direct access to MCP server or tools
  - Generate clinical report from ContextObject only
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 4.1 Write property test for synthesis completeness
  - **Property 15: Anomaly report content**
  - **Property 16: Report completeness**
  - **Validates: Requirements 5.4, 5.5**

- [x] 4.2 Implement Agent B LLM integration


  - Add Ollama client for MedGemma-27B model
  - Implement medical domain system prompt loading
  - Format ContextObject for LLM consumption
  - Parse LLM response into structured clinical report
  - _Requirements: 1.3, 9.2_


- [x] 4.3 Implement anomaly-aware synthesis





  - Add logic to detect model-knowledge discrepancies
  - Flag potential mixed pathology in report
  - Highlight SHAP-condition mismatches
  - Generate recommendations for multiple pathologies
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ]* 4.4 Write property test for anomaly awareness
  - **Property 17: Mixed pathology flagging**
  - **Property 18: Disease association listing**
  - **Property 19: SHAP-condition mismatch highlighting**
  - **Property 20: Multiple pathology recommendations**
  - **Validates: Requirements 6.1, 6.3, 6.4, 6.5**

-

- [x] 4.5 Implement counterfactual explanation




  - Add logic to interpret counterfactual results
  - Identify key diagnostic drivers based on confidence delta
  - Generate clinical explanations for feature impact
  - _Requirements: 7.2, 7.3, 7.4_

- [ ]* 4.6 Write property test for counterfactual interpretation
  - **Property 22: Significant confidence change identification**
  - **Property 23: Minimal confidence change indication**
  - **Validates: Requirements 7.3, 7.4**

---

## Phase 5: A2A Integration and Handoff

- [x] 5. Implement CDDAAgent with A2A coordination





  - Refactor `app/agents/cdda_agent.py` to use A2A pattern
  - Initialize DiagnosticMCPServer, Agent A, and Agent B
  - Implement `run_analysis(subject_id)` with handoff protocol
  - Agent A orchestrates → compiles ContextObject → hands off to Agent B
  - Agent B synthesizes → returns final report
  - _Requirements: 1.1, 1.2, 1.3, 3.1_

- [ ]* 5.1 Write property test for handoff protocol
  - **Property 27: Reasoning chain presence**
  - **Property 28: Reasoning chain structure**
  - **Validates: Requirements 8.3, 8.4**

- [x] 5.2 Implement reasoning chain aggregation (HIGH ROI - Required for paper)


  - Combine Agent A's reasoning with Agent B's reasoning
  - Include MCP actions in final reasoning chain with timestamps
  - Format reasoning chain for user display and paper evidence
  - Save complete reasoning trace to structured log file
  - _Requirements: 8.3, 8.4_
  - _Note: Essential for demonstrating agent collaboration in paper_

- [ ]* 5.3 Write property test for reasoning chain aggregation
  - **Property 28: Reasoning chain structure**
  - **Validates: Requirements 8.4**

---

## Phase 6: System Prompts and Configuration

- [x] 6. Create system prompt configuration files





  - Create `config/prompts/agent_a_orchestrator.txt` with MCP-aware prompt
  - Create `config/prompts/agent_b_consultant.txt` with medical domain prompt
  - Add prompt loading logic with hot-reload support
  - Add validation for prompt format and required sections
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 6.1 Create tool schema definitions


  - Create `config/schemas/mcp_tools.json` with tool schemas
  - Define JSON schema for simulate_counterfactual
  - Add schema validation logic
  - _Requirements: 9.5_

- [ ]* 6.2 Write property test for schema validation
  - **Property 30: Tool definition schema format**
  - **Validates: Requirements 9.5**

---

## Phase 7: Error Handling and Fallbacks

- [x] 7. Implement LLM error handling (HIGH ROI - Required for stability)





  - Add retry logic with exponential backoff for LLM calls (max 3 retries)
  - Implement timeout handling for slow LLM responses
  - Add JSON parsing error recovery for malformed function calls
  - Add try-except blocks around all LLM calls
  - Log all errors with context for debugging
  - _Requirements: 10.1_
  - _Note: Essential for demo stability, GPT-OSS may output malformed JSON_

- [ ]* 7.1 Write property test for retry logic
  - **Property 31: LLM retry with backoff**
  - **Validates: Requirements 10.1**

- [x] 7.2 Implement Agent A fallback logic


  - Add rule-based orchestration when Agent A LLM is unavailable
  - Use existing decision thresholds (UQ, anomaly)
  - Compile ContextObject from rule-based decisions
  - _Requirements: 10.2_

- [ ]* 7.3 Write property test for Agent A fallback
  - **Property 32: Orchestrator fallback**
  - **Validates: Requirements 10.2**



- [ ] 7.4 Implement Agent B fallback logic
  - Add template-based report generation when Agent B LLM is unavailable
  - Use existing synthesis methods from Phase 2
  - Ensure report completeness even in fallback mode
  - _Requirements: 10.3_

- [x]* 7.5 Write property test for Agent B fallback


  - **Property 33: Consultant fallback**
  - **Validates: Requirements 10.3**

- [ ] 7.6 Implement GraphRAG fallback
  - Ensure MCP server uses fallback knowledge base when GraphRAG fails
  - Add error annotations to ContextObject
  - _Requirements: 10.4_



- [ ]* 7.7 Write property test for GraphRAG fallback
  - **Property 13: GraphRAG fallback**
  - **Property 34: GraphRAG fallback**
  - **Validates: Requirements 4.5, 10.4**

- [ ] 7.8 Implement final fallback with error annotations
  - Add error tracking throughout the pipeline
  - Annotate final report with error information
  - Ensure system never fails completely
  - _Requirements: 10.5_

- [ ]* 7.9 Write property test for error annotations
  - **Property 29: Error logging**
  - **Property 35: Final fallback with annotations**
  - **Validates: Requirements 8.5, 10.5**

---

## Phase 8: Integration Testing and Validation

- [x] 8. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.


- [x] 8.1 Create end-to-end integration tests

  - Test standard case (low UQ, no anomalies)
  - Test high uncertainty case (triggers counterfactual)
  - Test anomaly case (triggers knowledge graph)
  - Test mixed case (both counterfactual and knowledge graph)
  - _Requirements: All_

- [x] 8.2 Create MCP compliance tests


  - Verify resource URIs follow MCP format
  - Verify tool schemas are valid JSON
  - Verify separation of resources and tools
  - Test MCP server with mock clients
  - _Requirements: 2.1, 2.2, 4.1, 4.2_

- [x] 8.3 Create A2A handoff tests


  - Verify ContextObject contains all required data
  - Verify Agent B has no tool access
  - Verify handoff protocol works correctly
  - Test handoff with various context sizes
  - _Requirements: 5.1, 8.3_

- [x] 8.4 Create fallback integration tests


  - Test system with Agent A unavailable
  - Test system with Agent B unavailable
  - Test system with GraphRAG unavailable
  - Test system with all LLMs unavailable
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

---

## Phase 9: Documentation and Demo

- [x] 9. Create comprehensive documentation





  - Update `docs/CDDA_Architecture_Spec.md` with MCP and A2A patterns
  - Create `docs/CDDA_Phase4_Complete.md` with implementation summary
  - Add MCP server API documentation
  - Add A2A handoff protocol documentation
  - _Requirements: All_

- [x] 9.1 Create demo scripts


  - Create `scripts/demo_mcp_server.py` demonstrating MCP interface
  - Create `scripts/demo_a2a_agents.py` demonstrating agent handoff
  - Create `scripts/demo_phase4_complete.py` demonstrating full system
  - Add example outputs for each demo
  - _Requirements: All_

- [x] 9.2 Update existing documentation


  - Update `CDDA_IMPLEMENTATION_STATUS.md` with Phase 4 status
  - Update `README.md` with Phase 4 usage instructions
  - Add troubleshooting guide for LLM issues
  - _Requirements: All_

---

## Phase 10: Final Checkpoint

- [x] 10. Final checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.
  - Verify MCP compliance
  - Verify A2A handoff protocol
  - Verify LLM integration
  - Verify fallback mechanisms
  - _Requirements: All_

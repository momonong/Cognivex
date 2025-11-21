# Requirements Document - CDDA Phase 4: Dual-LLM Integration

## Introduction

This specification defines Phase 4 of the Cognitive Discrepancy-Driven Agent (CDDA) framework: the integration of a dual-LLM architecture for autonomous clinical reasoning and report synthesis. The system will use two specialized LLMs working in tandem:

1. **Orchestrator LLM (GPT-OSS-20B)**: Handles function calling, decision logic, and tool orchestration
2. **Consultant LLM (MedGemma-27B)**: Performs final clinical synthesis and interpretation

This phase transforms the CDDA Agent from a rule-based system into an LLM-augmented autonomous diagnostic assistant capable of sophisticated clinical reasoning.

## Glossary

- **CDDA Agent**: Cognitive Discrepancy-Driven Agent - autonomous diagnostic system following MCP and A2A patterns
- **MCP (Model Context Protocol)**: Protocol that separates read-only data (Resources) from executable actions (Tools)
- **A2A (Agent-to-Agent)**: Pattern where specialized agents collaborate through structured handoffs
- **DiagnosticMCPServer**: Context layer providing MCP-compliant interface to diagnostic resources and tools
- **Agent A (Orchestrator)**: MCP client (GPT-OSS-20B) that reads resources, invokes tools, and compiles context
- **Agent B (Consultant)**: Medical specialist (MedGemma-27B) that synthesizes clinical reports from provided context
- **Resources**: Read-only data accessible via MCP (diagnostic reports, knowledge graph context)
- **Tools**: Executable actions accessible via MCP (counterfactual simulation)
- **ContextObject**: Structured data package handed off from Agent A to Agent B
- **Handoff**: Transfer of control and context from Agent A to Agent B
- **Tool Kit**: Collection of diagnostic tools (Layer 1+2) including RF prediction, SHAP, UQ, and Z-score analysis
- **GraphRAG**: Graph Retrieval-Augmented Generation service (Layer 4) for knowledge graph queries
- **UQ Score**: Uncertainty Quantification score (0.0-1.0) indicating prediction confidence
- **Anomaly Status**: Boolean flag indicating presence of statistical outliers in brain region measurements
- **Counterfactual Simulation**: What-if analysis masking specific features to assess their diagnostic impact
- **Knowledge Context**: Clinical information retrieved from Neo4j knowledge graph about anomalous regions
- **SHAP Values**: SHapley Additive exPlanations - feature importance scores from the ML model
- **Z-Scores**: Standardized scores indicating how many standard deviations a measurement is from population mean

## Requirements

### Requirement 1: Dual-LLM Initialization

**User Story:** As a system architect, I want the CDDA Agent to initialize two separate LLM instances with distinct roles following the Medical Expert-Assistant strategy, so that medical reasoning and function calling are handled by specialized models.

#### Acceptance Criteria

1. WHEN the CDDAAgent is initialized THEN the system SHALL create two separate Ollama LLM instances: Planner (MedGemma-27B) and Executor (GPT-OSS-20B)
2. WHEN initializing the Planner LLM THEN the system SHALL configure it with a medical domain system prompt including tool descriptions and decision logic
3. WHEN initializing the Executor LLM THEN the system SHALL configure it for function calling with JSON tool schemas
4. WHEN both LLMs are initialized THEN the system SHALL verify connectivity to the Ollama server
5. WHEN LLM initialization fails THEN the system SHALL provide clear error messages and fallback options

### Requirement 2: Command Translation and Function Calling

**User Story:** As a developer, I want the Executor LLM to translate the Planner's structured commands into executable function calls, so that the Planner can focus on medical reasoning without worrying about syntax details.

#### Acceptance Criteria

1. WHEN the Agent initializes THEN the system SHALL register all available tools with JSON schemas for the Executor LLM
2. WHEN the Planner LLM outputs a structured command THEN the system SHALL pass it to the Executor LLM for translation
3. WHEN the Executor LLM receives a command THEN it SHALL generate valid JSON function calls with proper parameters
4. WHEN a function call is generated THEN the system SHALL parse it and invoke the corresponding tool method
5. WHEN a tool returns results THEN the system SHALL format the results and pass them back to the Planner LLM

### Requirement 3: Autonomous Decision Flow

**User Story:** As a clinician, I want the CDDA Agent to autonomously analyze diagnostic data and decide which tools to invoke, so that I receive comprehensive analysis without manual intervention.

#### Acceptance Criteria

1. WHEN run_analysis is called THEN the system SHALL invoke get_diagnostic_report as the first step
2. WHEN the UQ score exceeds the threshold THEN the Planner LLM SHALL decide to invoke simulate_counterfactual and output a structured command
3. WHEN anomalies are detected THEN the Planner LLM SHALL decide to invoke query_knowledge_graph and output a structured command
4. WHEN neither condition is met THEN the Planner LLM SHALL proceed directly to report synthesis
5. WHEN the Planner makes a decision THEN the system SHALL log the medical reasoning chain for transparency

### Requirement 4: Knowledge Graph Integration

**User Story:** As a diagnostic system, I want to query the knowledge graph for clinical context about anomalous brain regions, so that I can provide evidence-based explanations for unusual patterns.

#### Acceptance Criteria

1. WHEN anomalous regions are identified THEN the system SHALL query GraphRAG for each region
2. WHEN querying GraphRAG THEN the system SHALL retrieve full name, function, clinical significance, and related conditions
3. WHEN multiple regions are anomalous THEN the system SHALL batch query for efficiency
4. WHEN GraphRAG returns results THEN the system SHALL format them as structured context for the Consultant LLM
5. WHEN GraphRAG is unavailable THEN the system SHALL use fallback knowledge base without failing

### Requirement 5: Clinical Report Synthesis

**User Story:** As a clinician, I want the Planner LLM to synthesize all diagnostic evidence into a coherent clinical report, so that I can quickly understand the diagnosis and its supporting evidence.

#### Acceptance Criteria

1. WHEN synthesis is triggered THEN the system SHALL pass all diagnostic data and tool results to the Planner LLM
2. WHEN the Planner LLM receives data THEN it SHALL integrate SHAP values, Z-scores, and knowledge context using medical expertise
3. WHEN generating the report THEN the Planner LLM SHALL explain the relationship between computational evidence and clinical knowledge
4. WHEN anomalies are present THEN the report SHALL explicitly address potential mixed pathology or atypical presentation
5. WHEN the report is complete THEN it SHALL include prediction, confidence, key findings, and clinical recommendations

### Requirement 6: Anomaly-Aware Synthesis

**User Story:** As a clinician, I want the system to highlight when model predictions conflict with knowledge graph evidence, so that I can identify potential mixed pathology or diagnostic uncertainty.

#### Acceptance Criteria

1. WHEN the model predicts AD with high confidence AND anomalous regions are associated with non-AD conditions THEN the report SHALL flag potential mixed pathology
2. WHEN knowledge graph context contradicts model prediction THEN the Planner LLM SHALL explain the discrepancy using medical reasoning
3. WHEN anomalous regions have known disease associations THEN the report SHALL list those associations
4. WHEN the leading SHAP feature is associated with a different condition THEN the report SHALL highlight this finding
5. WHEN multiple pathologies are suggested THEN the report SHALL recommend additional clinical correlation

### Requirement 7: Counterfactual Explanation

**User Story:** As a clinician, I want to understand which brain regions are driving the diagnosis, so that I can validate the model's reasoning against clinical findings.

#### Acceptance Criteria

1. WHEN high uncertainty is detected THEN the Planner LLM SHALL decide to simulate counterfactuals for top contributing features
2. WHEN counterfactual simulation completes THEN the Planner LLM SHALL explain the impact of masked features using medical reasoning
3. WHEN confidence changes significantly THEN the report SHALL identify those features as key diagnostic drivers
4. WHEN confidence changes minimally THEN the report SHALL indicate those features are not primary drivers
5. WHEN counterfactual results are presented THEN they SHALL include original vs. counterfactual predictions with confidence deltas

### Requirement 8: Reasoning Chain Transparency

**User Story:** As a system auditor, I want to see the complete reasoning chain from data to decision, so that I can verify the agent's logic and identify potential errors.

#### Acceptance Criteria

1. WHEN the Agent makes any decision THEN it SHALL log the decision and its justification
2. WHEN tools are invoked THEN the system SHALL record which tool was called and why
3. WHEN the analysis completes THEN the system SHALL provide a step-by-step reasoning chain
4. WHEN the reasoning chain is displayed THEN it SHALL show: data gathering, signal evaluation, tool selection, and synthesis
5. WHEN errors occur THEN they SHALL be included in the reasoning chain with recovery actions

### Requirement 9: System Prompt Management

**User Story:** As a developer, I want to define and manage system prompts for both LLMs, so that I can tune their behavior for optimal performance.

#### Acceptance Criteria

1. WHEN the Planner LLM is initialized THEN it SHALL receive a medical domain system prompt defining its role, available tools, decision logic, and synthesis guidelines
2. WHEN the Executor LLM is initialized THEN it SHALL receive a function-calling system prompt with JSON tool schemas
3. WHEN system prompts are defined THEN they SHALL be stored in configuration files for easy modification
4. WHEN prompts are updated THEN the system SHALL reload them without code changes
5. WHEN the Executor prompt includes tool definitions THEN they SHALL use JSON schema format for clarity

### Requirement 10: Error Handling and Fallback

**User Story:** As a system operator, I want the Agent to handle errors gracefully and continue operation, so that temporary failures don't prevent diagnosis.

#### Acceptance Criteria

1. WHEN an LLM call fails THEN the system SHALL retry with exponential backoff
2. WHEN the Planner LLM is unavailable THEN the system SHALL fall back to rule-based decision logic and template-based report generation
3. WHEN the Executor LLM is unavailable THEN the system SHALL directly invoke tools using parsed commands from the Planner
4. WHEN GraphRAG fails THEN the system SHALL use the fallback knowledge base
5. WHEN all fallbacks are exhausted THEN the system SHALL return a diagnostic report with error annotations

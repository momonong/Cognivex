# System Prompt Configuration Guide

## Overview

The CDDA Phase 4 system uses a centralized prompt configuration system with validation, hot-reload support, and tool schema definitions. This guide explains how to use and maintain the prompt system.

## Quick Start

### Loading Prompts

```python
from app.core.prompt_loader import PromptLoader

# Initialize loader
loader = PromptLoader()

# Load Agent A prompt
agent_a_prompt = loader.load_agent_a_prompt()

# Load Agent B prompt
agent_b_prompt = loader.load_agent_b_prompt()

# Load tool schemas
schemas = loader.load_tool_schemas()
```

### Using with Agents

```python
from app.agents.agent_a_orchestrator import AgentA
from app.agents.agent_b_consultant import AgentB
from app.core.mcp_server import DiagnosticMCPServer

# Agents automatically use PromptLoader
mcp_server = DiagnosticMCPServer()
agent_a = AgentA(mcp_server=mcp_server)
agent_b = AgentB()

# Prompts are loaded and validated automatically
```

## File Structure

```
config/
├── prompts/
│   ├── agent_a_orchestrator.txt    # Agent A system prompt
│   └── agent_b_consultant.txt      # Agent B system prompt
└── schemas/
    └── mcp_tools.json              # MCP tool and resource schemas
```

## Prompt Files

### Agent A Prompt (`agent_a_orchestrator.txt`)

**Purpose**: System prompt for the Orchestrator (GPT-OSS-20B)

**Required Sections**:
- Role description (Agent A, Orchestrator)
- MCP RESOURCES (read-only data)
- MCP TOOLS (executable actions)
- DECISION LOGIC (when to invoke tools)
- OUTPUT FORMAT (JSON structure)

**Validation**:
- Must contain all required sections
- Must include JSON output format example
- Must specify 'actions' and 'decision_rationale' fields

### Agent B Prompt (`agent_b_consultant.txt`)

**Purpose**: System prompt for the Clinical Consultant (MedGemma-27B)

**Required Sections**:
- Role description (Agent B, Clinical Consultant)
- INPUT description (ContextObject)
- SYNTHESIS GUIDELINES
- REPORT STRUCTURE

**Validation**:
- Must contain all required sections
- Must explicitly state "NO access to tools"
- Must describe ContextObject structure

## Tool Schema File

### MCP Tools Schema (`mcp_tools.json`)

**Structure**:
```json
{
  "resources": [
    {
      "uri": "diagnosis://{subject_id}/report",
      "name": "Diagnostic Report",
      "description": "...",
      "mime_type": "application/json",
      "parameters": { ... },
      "returns": { ... }
    }
  ],
  "tools": [
    {
      "name": "simulate_counterfactual",
      "description": "...",
      "parameters": { ... },
      "returns": { ... }
    }
  ]
}
```

**Validation**:
- Resources must have: uri, name, description, mime_type
- Resource URIs must start with `diagnosis://` or `knowledge://`
- Tools must have: name, description, parameters
- Parameters must be valid JSON Schema with type, properties, required

## Hot-Reload Support

The PromptLoader automatically detects file changes and reloads prompts:

```python
loader = PromptLoader()

# Load and cache
prompt1 = loader.load_agent_a_prompt()

# Edit config/prompts/agent_a_orchestrator.txt

# Load again (automatically detects change and reloads)
prompt2 = loader.load_agent_a_prompt()

# Or force reload
prompt3 = loader.load_agent_a_prompt(force_reload=True)
```

### Cache Management

```python
# Check cache status
cache_info = loader.get_cache_info()
print(cache_info)
# Output: {
#   'agent_a_cached': True,
#   'agent_b_cached': True,
#   'schemas_cached': True,
#   'cache_entries': 2,
#   'tracked_files': 3
# }

# Clear cache manually
loader.clear_cache()
```

## Validation

### Automatic Validation

All prompts and schemas are validated on load:

```python
try:
    prompt = loader.load_agent_a_prompt()
except ValueError as e:
    print(f"Validation failed: {e}")
    # Example: "Agent A prompt missing required sections: ['MCP TOOLS']"
```

### Manual Validation

You can validate prompts manually:

```python
# This is done automatically, but you can call it directly
loader._validate_agent_a_prompt(prompt_text)
loader._validate_agent_b_prompt(prompt_text)
loader._validate_tool_schemas(schemas)
```

## Error Handling

The system has multiple fallback levels:

1. **PromptLoader with validation** (preferred)
2. **Direct file read** (if PromptLoader fails)
3. **Embedded default prompt** (if file not found)

```python
# Agents handle fallbacks automatically
agent_a = AgentA(mcp_server=mcp_server)
# Will use PromptLoader, fall back to file read, then embedded prompt
```

## Modifying Prompts

### Editing Agent A Prompt

1. Open `config/prompts/agent_a_orchestrator.txt`
2. Make your changes
3. Ensure all required sections are present:
   - Agent A / Orchestrator
   - MCP RESOURCES
   - MCP TOOLS
   - DECISION LOGIC
   - OUTPUT FORMAT
4. Save the file
5. Next load will automatically use the new prompt

### Editing Agent B Prompt

1. Open `config/prompts/agent_b_consultant.txt`
2. Make your changes
3. Ensure all required sections are present:
   - Agent B / Clinical Consultant
   - INPUT / ContextObject
   - SYNTHESIS GUIDELINES
   - REPORT STRUCTURE
   - "NO access to tools" warning
4. Save the file
5. Next load will automatically use the new prompt

### Adding New Tools

1. Open `config/schemas/mcp_tools.json`
2. Add new tool to the `tools` array:
```json
{
  "name": "new_tool_name",
  "description": "What this tool does",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {
        "type": "string",
        "description": "Parameter description"
      }
    },
    "required": ["param1"]
  },
  "returns": {
    "type": "object",
    "properties": {
      "result": {"type": "string"}
    }
  }
}
```
3. Update Agent A prompt to mention the new tool
4. Implement the tool in `DiagnosticMCPServer.call_tool()`

### Adding New Resources

1. Open `config/schemas/mcp_tools.json`
2. Add new resource to the `resources` array:
```json
{
  "uri": "diagnosis://{subject_id}/new_resource",
  "name": "New Resource",
  "description": "What this resource provides",
  "mime_type": "application/json",
  "parameters": { ... },
  "returns": { ... }
}
```
3. Update Agent A prompt to mention the new resource
4. Implement the resource in `DiagnosticMCPServer.read_resource()`

## Testing

### Running Tests

```bash
# Test PromptLoader
python -m pytest tests/test_prompt_loader.py -v

# Test Agent integration
python -m pytest tests/test_agent_a_orchestrator.py::test_agent_a_initialization -v
python -m pytest tests/test_agent_b_consultant.py::test_agent_b_initialization -v

# Run all related tests
python -m pytest tests/test_prompt_loader.py tests/test_agent_a_orchestrator.py tests/test_agent_b_consultant.py -v
```

### Running Demo

```bash
# Comprehensive demo of prompt system
python demo_prompt_system.py

# PromptLoader demo only
python app/core/prompt_loader.py
```

## Troubleshooting

### Prompt Validation Fails

**Problem**: `ValueError: Agent A prompt missing required sections`

**Solution**: 
1. Check the error message for which sections are missing
2. Open the prompt file
3. Add the missing sections
4. Ensure section names match exactly (case-sensitive)

### File Not Found

**Problem**: `FileNotFoundError: Agent A prompt file not found`

**Solution**:
1. Verify file exists at `config/prompts/agent_a_orchestrator.txt`
2. Check file permissions
3. Ensure working directory is project root

### Schema Validation Fails

**Problem**: `ValueError: Tool schema missing required field: parameters`

**Solution**:
1. Open `config/schemas/mcp_tools.json`
2. Verify all tools have: name, description, parameters
3. Verify all resources have: uri, name, description, mime_type
4. Check JSON syntax is valid

### Cache Not Updating

**Problem**: Changes to prompt files not reflected

**Solution**:
```python
# Force reload
loader.load_agent_a_prompt(force_reload=True)

# Or clear cache
loader.clear_cache()
```

## Best Practices

### 1. Version Control
- Always commit prompt changes to git
- Use descriptive commit messages
- Review prompt changes in pull requests

### 2. Testing
- Test prompts after modifications
- Run validation tests before committing
- Use demo script to verify changes

### 3. Documentation
- Document prompt changes in commit messages
- Update this guide if adding new sections
- Keep schema documentation up to date

### 4. Validation
- Always validate prompts after editing
- Check all required sections are present
- Verify JSON format in schemas

### 5. Fallbacks
- Keep embedded prompts as last resort
- Test fallback behavior
- Log warnings when fallbacks are used

## API Reference

### PromptLoader Class

```python
class PromptLoader:
    def __init__(
        self,
        prompts_dir: str = "config/prompts",
        schemas_dir: str = "config/schemas"
    )
    
    # Loading methods
    def load_agent_a_prompt(self, force_reload: bool = False) -> str
    def load_agent_b_prompt(self, force_reload: bool = False) -> str
    def load_tool_schemas(self, force_reload: bool = False) -> Dict
    
    # Cache management
    def clear_cache(self) -> None
    def get_cache_info(self) -> Dict
    
    # Utilities
    def list_available_prompts(self) -> List[str]
    def list_available_schemas(self) -> List[str]
```

### Validation Methods

```python
# Internal validation methods (called automatically)
def _validate_agent_a_prompt(self, prompt_text: str) -> None
def _validate_agent_b_prompt(self, prompt_text: str) -> None
def _validate_tool_schemas(self, schemas: Dict) -> None
def _validate_resource_schema(self, resource: Dict) -> None
def _validate_tool_schema(self, tool: Dict) -> None
```

## Examples

### Example 1: Basic Usage

```python
from app.core.prompt_loader import PromptLoader

loader = PromptLoader()
agent_a_prompt = loader.load_agent_a_prompt()
print(f"Loaded prompt: {len(agent_a_prompt)} characters")
```

### Example 2: With Error Handling

```python
from app.core.prompt_loader import PromptLoader

loader = PromptLoader()

try:
    prompt = loader.load_agent_a_prompt()
    print("✓ Prompt loaded successfully")
except FileNotFoundError as e:
    print(f"✗ File not found: {e}")
except ValueError as e:
    print(f"✗ Validation failed: {e}")
```

### Example 3: Hot-Reload

```python
from app.core.prompt_loader import PromptLoader
import time

loader = PromptLoader()

# Load initial
prompt1 = loader.load_agent_a_prompt()
print(f"Initial: {len(prompt1)} chars")

# Simulate file modification
time.sleep(1)
# (Edit the file here)

# Load again (detects change)
prompt2 = loader.load_agent_a_prompt()
print(f"Reloaded: {len(prompt2)} chars")
```

### Example 4: Schema Inspection

```python
from app.core.prompt_loader import PromptLoader

loader = PromptLoader()
schemas = loader.load_tool_schemas()

print("Resources:")
for resource in schemas['resources']:
    print(f"  - {resource['uri']}")

print("\nTools:")
for tool in schemas['tools']:
    print(f"  - {tool['name']}")
    print(f"    Required params: {tool['parameters']['required']}")
```

## Related Documentation

- [CDDA Architecture Spec](CDDA_Architecture_Spec.md)
- [CDDA A2A Architecture](CDDA_A2A_ARCHITECTURE.md)
- [Task 6 Completion Summary](../TASK_6_COMPLETION_SUMMARY.md)
- [Agent A Model Setup](AGENT_A_MODEL_SETUP.md)

## Support

For issues or questions:
1. Check this guide first
2. Review test files for examples
3. Run demo script to verify setup
4. Check validation error messages

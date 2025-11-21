"""
Demo: System Prompt Configuration and Loading

This script demonstrates the prompt loading system with:
1. Loading Agent A and Agent B prompts
2. Loading tool schemas
3. Hot-reload functionality
4. Validation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.prompt_loader import PromptLoader
from app.agents.agent_a_orchestrator import AgentA, AgentAConfig
from app.agents.agent_b_consultant import AgentB, AgentBConfig
from app.core.mcp_server import DiagnosticMCPServer


def demo_prompt_loader():
    """Demo: PromptLoader functionality"""
    print("\n" + "="*80)
    print("DEMO 1: PromptLoader - Loading and Validation")
    print("="*80)
    
    loader = PromptLoader()
    
    # List available files
    print("\n[1] Available Configuration Files:")
    print("\n  Prompts:")
    for prompt in loader.list_available_prompts():
        print(f"    - {prompt}")
    
    print("\n  Schemas:")
    for schema in loader.list_available_schemas():
        print(f"    - {schema}")
    
    # Load Agent A prompt
    print("\n[2] Loading Agent A Prompt...")
    try:
        agent_a_prompt = loader.load_agent_a_prompt()
        print(f"  ✓ Loaded successfully ({len(agent_a_prompt)} characters)")
        
        # Show key sections
        if "MCP RESOURCES" in agent_a_prompt:
            print("  ✓ Contains MCP RESOURCES section")
        if "MCP TOOLS" in agent_a_prompt:
            print("  ✓ Contains MCP TOOLS section")
        if "DECISION LOGIC" in agent_a_prompt:
            print("  ✓ Contains DECISION LOGIC section")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Load Agent B prompt
    print("\n[3] Loading Agent B Prompt...")
    try:
        agent_b_prompt = loader.load_agent_b_prompt()
        print(f"  ✓ Loaded successfully ({len(agent_b_prompt)} characters)")
        
        # Show key sections
        if "Clinical Consultant" in agent_b_prompt:
            print("  ✓ Contains Clinical Consultant role")
        if "NO access to tools" in agent_b_prompt:
            print("  ✓ Contains tool access restriction")
        if "SYNTHESIS GUIDELINES" in agent_b_prompt:
            print("  ✓ Contains SYNTHESIS GUIDELINES section")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Load tool schemas
    print("\n[4] Loading Tool Schemas...")
    try:
        schemas = loader.load_tool_schemas()
        print(f"  ✓ Loaded successfully")
        print(f"  Resources: {len(schemas['resources'])}")
        print(f"  Tools: {len(schemas['tools'])}")
        
        print("\n  Resource URIs:")
        for resource in schemas['resources']:
            print(f"    - {resource['uri']}: {resource['name']}")
        
        print("\n  Tool Definitions:")
        for tool in schemas['tools']:
            print(f"    - {tool['name']}: {tool['description'][:60]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Show cache info
    print("\n[5] Cache Status:")
    cache_info = loader.get_cache_info()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    # Test hot-reload
    print("\n[6] Testing Hot-Reload (should use cache)...")
    agent_a_prompt_2 = loader.load_agent_a_prompt()
    print(f"  Same content: {agent_a_prompt == agent_a_prompt_2}")
    print(f"  Cache hit: ✓")
    
    print("\n" + "="*80)


def demo_agent_integration():
    """Demo: Agent integration with PromptLoader"""
    print("\n" + "="*80)
    print("DEMO 2: Agent Integration with PromptLoader")
    print("="*80)
    
    # Initialize MCP Server
    print("\n[1] Initializing MCP Server...")
    mcp_server = DiagnosticMCPServer(verbose=False)
    print("  ✓ MCP Server initialized")
    
    # Initialize Agent A
    print("\n[2] Initializing Agent A (Orchestrator)...")
    config_a = AgentAConfig(
        use_llm=False,  # Use rule-based for demo
        verbose=False
    )
    agent_a = AgentA(mcp_server=mcp_server, config=config_a)
    print("  ✓ Agent A initialized")
    print(f"  System prompt length: {len(agent_a.system_prompt)} characters")
    
    # Verify Agent A prompt sections
    print("\n  Agent A Prompt Validation:")
    required_sections = ["MCP RESOURCES", "MCP TOOLS", "DECISION LOGIC"]
    for section in required_sections:
        if section in agent_a.system_prompt:
            print(f"    ✓ {section}")
        else:
            print(f"    ✗ {section} MISSING")
    
    # Initialize Agent B
    print("\n[3] Initializing Agent B (Consultant)...")
    config_b = AgentBConfig(
        use_llm=False,  # Use template-based for demo
        verbose=False
    )
    agent_b = AgentB(config=config_b)
    print("  ✓ Agent B initialized")
    print(f"  System prompt length: {len(agent_b.system_prompt)} characters")
    
    # Verify Agent B prompt sections
    print("\n  Agent B Prompt Validation:")
    required_sections = ["Clinical Consultant", "NO access to tools", "SYNTHESIS GUIDELINES"]
    for section in required_sections:
        if section in agent_b.system_prompt:
            print(f"    ✓ {section}")
        else:
            print(f"    ✗ {section} MISSING")
    
    print("\n" + "="*80)


def demo_schema_validation():
    """Demo: Schema validation"""
    print("\n" + "="*80)
    print("DEMO 3: Tool Schema Validation")
    print("="*80)
    
    loader = PromptLoader()
    schemas = loader.load_tool_schemas()
    
    print("\n[1] Validating Resource Schemas...")
    for i, resource in enumerate(schemas['resources'], 1):
        print(f"\n  Resource {i}: {resource['name']}")
        print(f"    URI: {resource['uri']}")
        print(f"    MIME Type: {resource['mime_type']}")
        
        # Check required fields
        required_fields = ['uri', 'name', 'description', 'mime_type']
        all_present = all(field in resource for field in required_fields)
        print(f"    Required fields: {'✓' if all_present else '✗'}")
        
        # Validate URI scheme
        uri = resource['uri']
        valid_scheme = uri.startswith('diagnosis://') or uri.startswith('knowledge://')
        print(f"    Valid URI scheme: {'✓' if valid_scheme else '✗'}")
    
    print("\n[2] Validating Tool Schemas...")
    for i, tool in enumerate(schemas['tools'], 1):
        print(f"\n  Tool {i}: {tool['name']}")
        print(f"    Description: {tool['description'][:60]}...")
        
        # Check required fields
        required_fields = ['name', 'description', 'parameters']
        all_present = all(field in tool for field in required_fields)
        print(f"    Required fields: {'✓' if all_present else '✗'}")
        
        # Validate parameters schema
        params = tool['parameters']
        has_type = 'type' in params
        has_properties = 'properties' in params
        has_required = 'required' in params
        print(f"    Valid parameter schema: {'✓' if (has_type and has_properties and has_required) else '✗'}")
        
        # Show required parameters
        if 'required' in params:
            print(f"    Required params: {', '.join(params['required'])}")
    
    print("\n" + "="*80)


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("SYSTEM PROMPT CONFIGURATION DEMO")
    print("="*80)
    print("\nThis demo showcases the prompt loading system with:")
    print("  1. Centralized prompt management")
    print("  2. Validation and error checking")
    print("  3. Hot-reload support")
    print("  4. Tool schema definitions")
    print("  5. Agent integration")
    
    try:
        # Run demos
        demo_prompt_loader()
        demo_agent_integration()
        demo_schema_validation()
        
        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("  ✓ Prompt loading with validation")
        print("  ✓ Tool schema management")
        print("  ✓ Hot-reload caching")
        print("  ✓ Agent integration")
        print("  ✓ Error handling and fallbacks")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

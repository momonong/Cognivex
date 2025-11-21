"""
Tests for PromptLoader

This module tests prompt loading, validation, and hot-reload functionality.
"""

import pytest
import json
from pathlib import Path
from app.core.prompt_loader import PromptLoader


class TestPromptLoader:
    """Test suite for PromptLoader"""
    
    def test_load_agent_a_prompt(self):
        """Test loading Agent A prompt"""
        loader = PromptLoader()
        prompt = loader.load_agent_a_prompt()
        
        # Check prompt is not empty
        assert len(prompt) > 0
        
        # Check required sections are present
        assert "Agent A" in prompt
        assert "Orchestrator" in prompt
        assert "MCP RESOURCES" in prompt
        assert "MCP TOOLS" in prompt
        assert "DECISION LOGIC" in prompt
        assert "OUTPUT FORMAT" in prompt
    
    def test_load_agent_b_prompt(self):
        """Test loading Agent B prompt"""
        loader = PromptLoader()
        prompt = loader.load_agent_b_prompt()
        
        # Check prompt is not empty
        assert len(prompt) > 0
        
        # Check required sections are present
        assert "Agent B" in prompt
        assert "Clinical Consultant" in prompt
        assert "INPUT" in prompt
        assert "ContextObject" in prompt
        assert "SYNTHESIS GUIDELINES" in prompt
        assert "REPORT STRUCTURE" in prompt
        assert "NO access to tools" in prompt
    
    def test_load_tool_schemas(self):
        """Test loading tool schemas"""
        loader = PromptLoader()
        schemas = loader.load_tool_schemas()
        
        # Check schema structure
        assert "resources" in schemas
        assert "tools" in schemas
        
        # Check resources
        assert len(schemas["resources"]) > 0
        for resource in schemas["resources"]:
            assert "uri" in resource
            assert "name" in resource
            assert "description" in resource
            assert "mime_type" in resource
        
        # Check tools
        assert len(schemas["tools"]) > 0
        for tool in schemas["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
    
    def test_hot_reload_caching(self):
        """Test hot-reload caching mechanism"""
        loader = PromptLoader()
        
        # Load prompt first time
        prompt1 = loader.load_agent_a_prompt()
        
        # Load again (should use cache)
        prompt2 = loader.load_agent_a_prompt()
        
        # Should be identical
        assert prompt1 == prompt2
        
        # Check cache info
        cache_info = loader.get_cache_info()
        assert cache_info["agent_a_cached"] is True
    
    def test_force_reload(self):
        """Test force reload functionality"""
        loader = PromptLoader()
        
        # Load and cache
        prompt1 = loader.load_agent_a_prompt()
        
        # Force reload
        prompt2 = loader.load_agent_a_prompt(force_reload=True)
        
        # Should still be identical (same file)
        assert prompt1 == prompt2
    
    def test_clear_cache(self):
        """Test cache clearing"""
        loader = PromptLoader()
        
        # Load and cache
        loader.load_agent_a_prompt()
        loader.load_agent_b_prompt()
        loader.load_tool_schemas()
        
        # Verify cached
        cache_info = loader.get_cache_info()
        assert cache_info["agent_a_cached"] is True
        assert cache_info["agent_b_cached"] is True
        assert cache_info["schemas_cached"] is True
        
        # Clear cache
        loader.clear_cache()
        
        # Verify cleared
        cache_info = loader.get_cache_info()
        assert cache_info["agent_a_cached"] is False
        assert cache_info["agent_b_cached"] is False
        assert cache_info["schemas_cached"] is False
    
    def test_list_available_files(self):
        """Test listing available prompt and schema files"""
        loader = PromptLoader()
        
        # List prompts
        prompts = loader.list_available_prompts()
        assert "agent_a_orchestrator.txt" in prompts
        assert "agent_b_consultant.txt" in prompts
        
        # List schemas
        schemas = loader.list_available_schemas()
        assert "mcp_tools.json" in schemas
    
    def test_schema_validation(self):
        """Test schema validation"""
        loader = PromptLoader()
        schemas = loader.load_tool_schemas()
        
        # Validate resource URIs
        for resource in schemas["resources"]:
            uri = resource["uri"]
            assert uri.startswith("diagnosis://") or uri.startswith("knowledge://")
        
        # Validate tool parameters
        for tool in schemas["tools"]:
            params = tool["parameters"]
            assert "type" in params
            assert "properties" in params
            assert "required" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

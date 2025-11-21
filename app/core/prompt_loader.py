"""
CDDA Framework - Prompt Loader

This module provides utilities for loading and validating system prompts
for Agent A (Orchestrator) and Agent B (Consultant) with hot-reload support.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import re


class PromptLoader:
    """
    Loader for system prompts with validation and hot-reload support
    
    Features:
    - Load prompts from configuration files
    - Validate prompt format and required sections
    - Hot-reload support (check for file changes)
    - Load tool schemas for Agent A
    """
    
    def __init__(
        self,
        prompts_dir: str = "config/prompts",
        schemas_dir: str = "config/schemas"
    ):
        """
        Initialize PromptLoader
        
        Args:
            prompts_dir: Directory containing prompt files
            schemas_dir: Directory containing schema files
        """
        self.prompts_dir = Path(prompts_dir)
        self.schemas_dir = Path(schemas_dir)
        
        # Cache for loaded prompts and schemas
        self._prompt_cache: Dict[str, Dict] = {}
        self._schema_cache: Optional[Dict] = None
        
        # Track file modification times for hot-reload
        self._file_mtimes: Dict[str, float] = {}
    
    # ========================================================================
    # Prompt Loading
    # ========================================================================
    
    def load_agent_a_prompt(self, force_reload: bool = False) -> str:
        """
        Load Agent A (Orchestrator) system prompt
        
        Args:
            force_reload: Force reload even if cached
        
        Returns:
            System prompt text
        
        Raises:
            FileNotFoundError: If prompt file not found
            ValueError: If prompt validation fails
        """
        prompt_file = self.prompts_dir / "agent_a_orchestrator.txt"
        
        # Check if reload needed
        if not force_reload and self._is_cached("agent_a", prompt_file):
            return self._prompt_cache["agent_a"]["content"]
        
        # Load prompt
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Agent A prompt file not found: {prompt_file}"
            )
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        
        # Validate prompt
        self._validate_agent_a_prompt(prompt_text)
        
        # Cache prompt
        self._prompt_cache["agent_a"] = {
            "content": prompt_text,
            "loaded_at": datetime.now().isoformat()
        }
        self._file_mtimes[str(prompt_file)] = prompt_file.stat().st_mtime
        
        return prompt_text
    
    def load_agent_b_prompt(self, force_reload: bool = False) -> str:
        """
        Load Agent B (Consultant) system prompt
        
        Args:
            force_reload: Force reload even if cached
        
        Returns:
            System prompt text
        
        Raises:
            FileNotFoundError: If prompt file not found
            ValueError: If prompt validation fails
        """
        prompt_file = self.prompts_dir / "agent_b_consultant.txt"
        
        # Check if reload needed
        if not force_reload and self._is_cached("agent_b", prompt_file):
            return self._prompt_cache["agent_b"]["content"]
        
        # Load prompt
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Agent B prompt file not found: {prompt_file}"
            )
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        
        # Validate prompt
        self._validate_agent_b_prompt(prompt_text)
        
        # Cache prompt
        self._prompt_cache["agent_b"] = {
            "content": prompt_text,
            "loaded_at": datetime.now().isoformat()
        }
        self._file_mtimes[str(prompt_file)] = prompt_file.stat().st_mtime
        
        return prompt_text
    
    # ========================================================================
    # Schema Loading
    # ========================================================================
    
    def load_tool_schemas(self, force_reload: bool = False) -> Dict:
        """
        Load MCP tool schemas
        
        Args:
            force_reload: Force reload even if cached
        
        Returns:
            Dictionary with 'resources' and 'tools' schemas
        
        Raises:
            FileNotFoundError: If schema file not found
            ValueError: If schema validation fails
        """
        schema_file = self.schemas_dir / "mcp_tools.json"
        
        # Check if reload needed
        if not force_reload and self._is_schema_cached(schema_file):
            return self._schema_cache
        
        # Load schema
        if not schema_file.exists():
            raise FileNotFoundError(
                f"Tool schema file not found: {schema_file}"
            )
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schemas = json.load(f)
        
        # Validate schema
        self._validate_tool_schemas(schemas)
        
        # Cache schema
        self._schema_cache = schemas
        self._file_mtimes[str(schema_file)] = schema_file.stat().st_mtime
        
        return schemas
    
    # ========================================================================
    # Validation Methods
    # ========================================================================
    
    def _validate_agent_a_prompt(self, prompt_text: str) -> None:
        """
        Validate Agent A prompt format and required sections
        
        Required sections:
        - Role description
        - MCP RESOURCES
        - MCP TOOLS
        - DECISION LOGIC
        - OUTPUT FORMAT
        
        Args:
            prompt_text: Prompt text to validate
        
        Raises:
            ValueError: If validation fails
        """
        required_sections = [
            "Agent A",
            "Orchestrator",
            "MCP RESOURCES",
            "MCP TOOLS",
            "DECISION LOGIC",
            "OUTPUT FORMAT"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in prompt_text:
                missing_sections.append(section)
        
        if missing_sections:
            raise ValueError(
                f"Agent A prompt missing required sections: {missing_sections}"
            )
        
        # Validate JSON format example is present
        if "actions" not in prompt_text or "decision_rationale" not in prompt_text:
            raise ValueError(
                "Agent A prompt must include JSON output format example "
                "with 'actions' and 'decision_rationale' fields"
            )
    
    def _validate_agent_b_prompt(self, prompt_text: str) -> None:
        """
        Validate Agent B prompt format and required sections
        
        Required sections:
        - Role description
        - INPUT description
        - SYNTHESIS GUIDELINES
        - REPORT STRUCTURE
        
        Args:
            prompt_text: Prompt text to validate
        
        Raises:
            ValueError: If validation fails
        """
        required_sections = [
            "Agent B",
            "Clinical Consultant",
            "INPUT",
            "ContextObject",
            "SYNTHESIS GUIDELINES",
            "REPORT STRUCTURE"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in prompt_text:
                missing_sections.append(section)
        
        if missing_sections:
            raise ValueError(
                f"Agent B prompt missing required sections: {missing_sections}"
            )
        
        # Validate no tool access warning is present
        if "NO access to tools" not in prompt_text:
            raise ValueError(
                "Agent B prompt must explicitly state 'NO access to tools'"
            )
    
    def _validate_tool_schemas(self, schemas: Dict) -> None:
        """
        Validate tool schema structure
        
        Args:
            schemas: Schema dictionary to validate
        
        Raises:
            ValueError: If validation fails
        """
        # Check top-level structure
        if "resources" not in schemas:
            raise ValueError("Tool schemas missing 'resources' section")
        if "tools" not in schemas:
            raise ValueError("Tool schemas missing 'tools' section")
        
        # Validate resources
        for resource in schemas["resources"]:
            self._validate_resource_schema(resource)
        
        # Validate tools
        for tool in schemas["tools"]:
            self._validate_tool_schema(tool)
    
    def _validate_resource_schema(self, resource: Dict) -> None:
        """
        Validate individual resource schema
        
        Args:
            resource: Resource schema to validate
        
        Raises:
            ValueError: If validation fails
        """
        required_fields = ["uri", "name", "description", "mime_type"]
        
        for field in required_fields:
            if field not in resource:
                raise ValueError(
                    f"Resource schema missing required field: {field}"
                )
        
        # Validate URI format
        uri = resource["uri"]
        if not (uri.startswith("diagnosis://") or uri.startswith("knowledge://")):
            raise ValueError(
                f"Invalid resource URI scheme: {uri}. "
                f"Must start with 'diagnosis://' or 'knowledge://'"
            )
    
    def _validate_tool_schema(self, tool: Dict) -> None:
        """
        Validate individual tool schema
        
        Args:
            tool: Tool schema to validate
        
        Raises:
            ValueError: If validation fails
        """
        required_fields = ["name", "description", "parameters"]
        
        for field in required_fields:
            if field not in tool:
                raise ValueError(
                    f"Tool schema missing required field: {field}"
                )
        
        # Validate parameters is a JSON schema
        params = tool["parameters"]
        if not isinstance(params, dict):
            raise ValueError("Tool parameters must be a dictionary")
        
        if "type" not in params:
            raise ValueError("Tool parameters must have 'type' field")
        
        if "properties" not in params:
            raise ValueError("Tool parameters must have 'properties' field")
    
    # ========================================================================
    # Cache Management
    # ========================================================================
    
    def _is_cached(self, agent_key: str, prompt_file: Path) -> bool:
        """
        Check if prompt is cached and file hasn't changed
        
        Args:
            agent_key: Cache key ('agent_a' or 'agent_b')
            prompt_file: Path to prompt file
        
        Returns:
            True if cached and up-to-date
        """
        if agent_key not in self._prompt_cache:
            return False
        
        file_path = str(prompt_file)
        if file_path not in self._file_mtimes:
            return False
        
        # Check if file has been modified
        current_mtime = prompt_file.stat().st_mtime
        cached_mtime = self._file_mtimes[file_path]
        
        return current_mtime == cached_mtime
    
    def _is_schema_cached(self, schema_file: Path) -> bool:
        """
        Check if schema is cached and file hasn't changed
        
        Args:
            schema_file: Path to schema file
        
        Returns:
            True if cached and up-to-date
        """
        if self._schema_cache is None:
            return False
        
        file_path = str(schema_file)
        if file_path not in self._file_mtimes:
            return False
        
        # Check if file has been modified
        current_mtime = schema_file.stat().st_mtime
        cached_mtime = self._file_mtimes[file_path]
        
        return current_mtime == cached_mtime
    
    def clear_cache(self) -> None:
        """Clear all cached prompts and schemas"""
        self._prompt_cache.clear()
        self._schema_cache = None
        self._file_mtimes.clear()
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_cache_info(self) -> Dict:
        """
        Get information about cached prompts and schemas
        
        Returns:
            Dictionary with cache status
        """
        return {
            "agent_a_cached": "agent_a" in self._prompt_cache,
            "agent_b_cached": "agent_b" in self._prompt_cache,
            "schemas_cached": self._schema_cache is not None,
            "cache_entries": len(self._prompt_cache),
            "tracked_files": len(self._file_mtimes)
        }
    
    def list_available_prompts(self) -> List[str]:
        """
        List available prompt files
        
        Returns:
            List of prompt file names
        """
        if not self.prompts_dir.exists():
            return []
        
        return [f.name for f in self.prompts_dir.glob("*.txt")]
    
    def list_available_schemas(self) -> List[str]:
        """
        List available schema files
        
        Returns:
            List of schema file names
        """
        if not self.schemas_dir.exists():
            return []
        
        return [f.name for f in self.schemas_dir.glob("*.json")]


# ============================================================================
# Demo Functions
# ============================================================================

def demo_prompt_loading():
    """Demo: Load and validate prompts"""
    print("\n" + "="*80)
    print("DEMO: Prompt Loader - Loading and Validation")
    print("="*80)
    
    loader = PromptLoader()
    
    # List available files
    print("\n[1] Available Prompts:")
    for prompt in loader.list_available_prompts():
        print(f"  - {prompt}")
    
    print("\n[2] Available Schemas:")
    for schema in loader.list_available_schemas():
        print(f"  - {schema}")
    
    # Load Agent A prompt
    print("\n[3] Loading Agent A Prompt...")
    try:
        agent_a_prompt = loader.load_agent_a_prompt()
        print(f"  ✓ Loaded successfully ({len(agent_a_prompt)} characters)")
        print(f"  First 100 chars: {agent_a_prompt[:100]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Load Agent B prompt
    print("\n[4] Loading Agent B Prompt...")
    try:
        agent_b_prompt = loader.load_agent_b_prompt()
        print(f"  ✓ Loaded successfully ({len(agent_b_prompt)} characters)")
        print(f"  First 100 chars: {agent_b_prompt[:100]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Load tool schemas
    print("\n[5] Loading Tool Schemas...")
    try:
        schemas = loader.load_tool_schemas()
        print(f"  ✓ Loaded successfully")
        print(f"  Resources: {len(schemas['resources'])}")
        print(f"  Tools: {len(schemas['tools'])}")
        
        print("\n  Resource URIs:")
        for resource in schemas['resources']:
            print(f"    - {resource['uri']}")
        
        print("\n  Tool Names:")
        for tool in schemas['tools']:
            print(f"    - {tool['name']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Show cache info
    print("\n[6] Cache Info:")
    cache_info = loader.get_cache_info()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    # Test hot-reload (load again without force)
    print("\n[7] Testing Hot-Reload (should use cache)...")
    agent_a_prompt_2 = loader.load_agent_a_prompt()
    print(f"  Same content: {agent_a_prompt == agent_a_prompt_2}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demo_prompt_loading()

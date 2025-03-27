import os
import yaml
import json
import re
from typing import Dict, Any, Optional

def load_config_with_env_vars(config_path: str) -> Dict[str, Any]:
    """
    Load configuration file with environment variable substitution.
    Handles both YAML and JSON formats.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the configuration with environment variables substituted
    """
    if not config_path or not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config_text = f.read()
        
        # Function to replace ${VAR} with environment variable value
        def replace_env_var(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                print(f"WARNING: Environment variable '{var_name}' not found!")
                return "${" + var_name + "}"
            return value
        
        # Replace all ${VAR} patterns with their environment values
        config_text = re.sub(r'\${([^}]+)}', replace_env_var, config_text)
        
        # Load as YAML or JSON depending on file extension
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(config_text)
        else:
            config = json.loads(config_text)
        
        return config
    except Exception as e:
        import traceback
        print(f"Error loading config from {config_path}: {str(e)}")
        print(traceback.format_exc())
        return {}

def get_nested_config(config: Dict[str, Any], *keys, default=None) -> Any:
    """
    Safely get a nested value from a configuration dictionary.
    
    Args:
        config: The configuration dictionary
        *keys: The sequence of keys to navigate through
        default: Default value to return if the path doesn't exist
        
    Returns:
        The value at the specified path or the default value
    """
    result = config
    for key in keys:
        if not isinstance(result, dict) or key not in result:
            return default
        result = result[key]
    return result
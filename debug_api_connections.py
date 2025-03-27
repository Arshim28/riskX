import os
import sys
import json
import asyncio
import traceback
from dotenv import load_dotenv
from utils.config_utils import load_config_with_env_vars

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our project
from utils.llm_provider import init_llm_provider, get_llm_provider, LLMProvider
from utils.logging import setup_logging, get_logger

async def test_provider(provider_name, model_name, llm_provider):
    """Test a specific LLM provider with a simple query"""
    print(f"\n=== Testing {provider_name.upper()} Provider with model {model_name} ===")
    try:
        # Test model creation first
        print(f"Creating LLM instance...")
        llm = await llm_provider.get_llm(provider=provider_name, model=model_name)
        print(f"✓ Successfully created LLM instance")
        
        # Try a simple query
        print(f"Sending test query to {provider_name}...")
        response = await llm_provider.generate_text(
            "Hello! Please respond with a single short sentence to confirm you're working.",
            provider=provider_name,
            model=model_name
        )
        print(f"✓ Received response: {response[:100]}" + ("..." if len(response) > 100 else ""))
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        print(f"Detailed error: {traceback.format_exc()}")
        return False

async def debug_api_connections(config_path):
    """Test all configured API connections"""
    # Setup logging
    setup_logging("api_debug")
    logger = get_logger("api_debug")
    
    print(f"Loading config from: {config_path}")
    try:
        # Load config with proper environment variable substitution
        config = load_config_with_env_vars(config_path)
        
        # Display environment variable status
        print("\n=== Environment Variables ===")
        for key in ['YOUTUBE_API_KEY', 'GOOGLE_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'SERPAPI_API_KEY']:
            value = os.environ.get(key)
            masked_value = value[:4] + "..." + value[-4:] if value and len(value) > 10 else "(not set)"
            print(f"{key}: {'✓ Found' if value else '✗ Missing'} - {masked_value}")
        
        # Display config details
        print("\n=== Config File Analysis ===")
        if 'llm_provider' in config:
            llm_config = config['llm_provider']
            print(f"Default provider: {llm_config.get('default_provider', 'Not specified')}")
            print(f"Configured providers: {list(llm_config.get('providers', {}).keys())}")
            
            # Check for API keys in config
            providers = llm_config.get('providers', {})
            for provider, settings in providers.items():
                api_key = settings.get('api_key')
                if api_key and isinstance(api_key, str):
                    if api_key.startswith('${') and api_key.endswith('}'):
                        print(f"✗ {provider.upper()} API key is using environment variable syntax: {api_key}")
                    else:
                        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
                        print(f"✓ {provider.upper()} API key found in config: {masked_key}")
                else:
                    print(f"✗ {provider.upper()} API key missing from config")
        else:
            print("✗ llm_provider section not found in config!")
            
        # Initialize LLM provider
        print("\n=== Initializing LLM Provider ===")
        llm_provider = init_llm_provider(config)
        
        # Check provider initialization status
        print("\n=== LLM Provider Configuration ===")
        default_provider = llm_provider.config.default_provider
        configured_providers = []
        
        for provider_name, provider_config in llm_provider.config.providers.items():
            api_key = provider_config.api_key
            default_model = provider_config.default_model
            if api_key:
                masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
                print(f"✓ {provider_name.upper()}: API Key found ({masked_key}), Default model: {default_model}")
                configured_providers.append(provider_name)
            else:
                print(f"✗ {provider_name.upper()}: No API key available, Default model: {default_model}")
        
        print(f"\nDefault provider: {default_provider}")
        print(f"Configured providers: {configured_providers}")
        
        # Test each configured provider
        print("\n=== Testing API Connections ===")
        results = []
        
        for provider_name in configured_providers:
            provider_config = llm_provider.config.providers[provider_name]
            model = provider_config.default_model
            success = await test_provider(provider_name, model, llm_provider)
            results.append((provider_name, model, success))
            
        # Test SerpAPI if available
        serpapi_key = config.get('research', {}).get('api_key')
        if serpapi_key:
            print("\n=== Testing SerpAPI ===")
            import requests
            try:
                url = f"https://serpapi.com/search?q=test&api_key={serpapi_key}&engine=google"
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"✓ SerpAPI connection successful")
                    results.append(("serpapi", "N/A", True))
                else:
                    print(f"✗ SerpAPI error: {response.status_code} - {response.text}")
                    results.append(("serpapi", "N/A", False))
            except Exception as e:
                print(f"✗ SerpAPI error: {str(e)}")
                results.append(("serpapi", "N/A", False))
        
        # Test YouTube API if available
        youtube_key = config.get('youtube', {}).get('youtube_api_key')
        if youtube_key:
            print("\n=== Testing YouTube API ===")
            import requests
            try:
                url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q=test&key={youtube_key}&maxResults=1"
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"✓ YouTube API connection successful")
                    results.append(("youtube", "N/A", True))
                else:
                    print(f"✗ YouTube API error: {response.status_code} - {response.text}")
                    results.append(("youtube", "N/A", False))
            except Exception as e:
                print(f"✗ YouTube API error: {str(e)}")
                results.append(("youtube", "N/A", False))
        
        # Summary
        print("\n=== Results Summary ===")
        all_success = True
        for name, model, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            model_info = f" ({model})" if model != "N/A" else ""
            print(f"{name.upper()}{model_info}: {status}")
            if not success:
                all_success = False
        
        if all_success:
            print("\nAll API connections successful! 🎉")
        else:
            print("\nSome API connections failed. Check the errors above.")
            
        print("\n=== Suggested Fixes ===")
        print("1. Make sure your .env file is in the correct location (project root)")
        print("2. Verify .env format is correct (no spaces around = sign)")
        print("3. Check for typos in API keys")
        print("4. Try replacing ${ENV_VAR} syntax with direct values in config.yaml")
        print("5. Run with environment variables set directly: GOOGLE_API_KEY=your_key python script.py")
        
    except Exception as e:
        print(f"Error during API connection debugging: {str(e)}")
        print(traceback.format_exc())

async def add_config_utils():
    """Create config_utils.py if it doesn't exist"""
    utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
    config_utils_path = os.path.join(utils_dir, "config_utils.py")
    
    if not os.path.exists(utils_dir):
        os.makedirs(utils_dir)
        
    if not os.path.exists(config_utils_path):
        print("Creating config_utils.py...")
        with open(config_utils_path, 'w') as f:
            f.write("""import os
import yaml
import json
import re
from typing import Dict, Any, Optional

def load_config_with_env_vars(config_path: str) -> Dict[str, Any]:
    \"\"\"
    Load configuration file with environment variable substitution.
    Handles both YAML and JSON formats.
    \"\"\"
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
        config_text = re.sub(r'\\${([^}]+)}', replace_env_var, config_text)
        
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
    \"\"\"
    Safely get a nested value from a configuration dictionary.
    \"\"\"
    result = config
    for key in keys:
        if not isinstance(result, dict) or key not in result:
            return default
        result = result[key]
    return result
""")
        print("✓ Created config_utils.py")

if __name__ == "__main__":
    load_dotenv(dotenv_path='.env', override=True)
    
    # Create config_utils.py if needed
    
    # Get config path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    
    # Run the debug
    asyncio.run(debug_api_connections(config_path))
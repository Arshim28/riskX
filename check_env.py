import os
import yaml
import re
from dotenv import load_dotenv

def load_config_with_env_vars(config_path):
    load_dotenv(dotenv_path='.env', override=True)
    
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    def replace_env_var(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            print(f"WARNING: Environment variable {var_name} not found!")
            return "${" + var_name + "}"
        return value
    
    config_text = re.sub(r'\${([^}]+)}', replace_env_var, config_text)
    config = yaml.safe_load(config_text)
    
    return config

def check_api_keys(config):
    youtube_key = config.get('youtube', {}).get('youtube_api_key')
    if youtube_key and youtube_key.startswith('${'):
        print(f"ERROR: YouTube API key not resolved: {youtube_key}")
    else:
        print(f"YouTube API key: {'✓ Found' if youtube_key else '✗ Missing'}")
    
    llm_providers = config.get('llm_provider', {}).get('providers', {})
    
    for provider, settings in llm_providers.items():
        api_key = settings.get('api_key')
        if api_key and api_key.startswith('${'):
            print(f"ERROR: {provider.upper()} API key not resolved: {api_key}")
        else:
            print(f"{provider.upper()} API key: {'✓ Found' if api_key else '✗ Missing'}")
    
    serp_key = config.get('research', {}).get('api_key')
    if serp_key and serp_key.startswith('${'):
        print(f"ERROR: SERPAPI API key not resolved: {serp_key}")
    else:
        print(f"SERPAPI API key: {'✓ Found' if serp_key else '✗ Missing'}")

config = load_config_with_env_vars('config.yaml')
check_api_keys(config)

print("\nEnvironment variables:")
for key in ['YOUTUBE_API_KEY', 'GOOGLE_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'SERPAPI_API_KEY']:
    value = os.environ.get(key)
    print(f"{key}: {'✓ Found' if value else '✗ Missing'}")
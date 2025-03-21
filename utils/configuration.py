import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

DEFAULT_CONFIG = {
    "chunk_size": 10000,
    "chunk_overlap": 500,
    "index_type": "Flat",
    "embedding_dimension": 3072,
    "max_tokens": 8000,
    "log_level": "INFO",
    
    "retry_max_attempts": 5,
    "retry_base_delay": 1.0,
    "request_delay": 0.5,
}

def load_config(config_path: str = "../config.yaml") -> Dict[str, Any]:
    
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            if yaml_config.get("document_processor"):
                doc_config = yaml_config["document_processor"]
                config["chunk_size"] = doc_config.get("chunk_size", config["chunk_size"])
                config["chunk_overlap"] = doc_config.get("chunk_overlap", config["chunk_overlap"])
                
            if yaml_config.get("embedding"):
                embed_config = yaml_config["embedding"]
                config["embedding_dimension"] = embed_config.get("dimension", config["embedding_dimension"])
                config["max_tokens"] = embed_config.get("max_tokens", config["max_tokens"])
                config["request_delay"] = embed_config.get("request_delay", config["request_delay"])
                config["retry_max_attempts"] = embed_config.get("retry_max_attempts", config["retry_max_attempts"])
                config["retry_base_delay"] = embed_config.get("retry_base_delay", config["retry_base_delay"])
                
            if yaml_config.get("vector_store"):
                vector_config = yaml_config["vector_store"]
                config["index_type"] = vector_config.get("index_type", config["index_type"])
                
            if yaml_config.get("app"):
                app_config = yaml_config["app"]
                config["log_level"] = app_config.get("log_level", config["log_level"])
                
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration")
    
    for key in config:
        env_var = key.upper()
        if os.environ.get(env_var):
            if isinstance(config[key], int):
                config[key] = int(os.environ.get(env_var))
            elif isinstance(config[key], float):
                config[key] = float(os.environ.get(env_var))
            else:
                config[key] = os.environ.get(env_var)
    
    return config


def validate_config():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable is required")
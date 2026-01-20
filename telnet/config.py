"""Configuration loading and management."""

import os
import re
from pathlib import Path
from typing import Dict, Any
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable expansion.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    config = _expand_env_vars(config)
    
    return config


def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand environment variables in configuration.
    
    Supports format: ${VAR_NAME:default_value}
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) or ''
            return os.environ.get(var_name, default)
        
        return re.sub(pattern, replacer, obj)
    else:
        return obj


def save_config(config: Dict[str, Any], path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        'region': {
            'name': 'default',
            'lat_min': -60.0,
            'lat_max': 15.0,
            'lon_min': -85.0,
            'lon_max': -30.0,
        },
        'model': {
            'nunits': 128,
            'dropout': 0.2,
            'epochs': 200,
            'learning_rate': 0.001,
            'batch_size': 24,
            'nfeats': 10,
        },
        'training': {
            'seed': 42,
            'val_split': 0.2,
        },
    }

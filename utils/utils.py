import json
import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import tempfile
import aiofiles


async def load_yaml_config(file_path: str) -> Dict[str, Any]:
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    return yaml.safe_load(content)


async def save_to_file(content: str, file_path: Union[str, Path], mode: str = 'w') -> str:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiofiles.open(file_path, mode, encoding='utf-8') as f:
        await f.write(content)
    
    return str(file_path)


async def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> str:
    content = json.dumps(data, indent=2)
    return await save_to_file(content, file_path)


async def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        content = await f.read()
    
    return json.loads(content)


async def create_temp_file(content: str, suffix: str = '.txt') -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(content)
        return path
    except Exception:
        os.close(fd)
        os.unlink(path)
        raise


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in name)
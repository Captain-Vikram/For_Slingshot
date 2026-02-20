# Minimal TOON decoder for the object detection schema
import re
from typing import Any

def decode(text: str) -> dict[str, Any] | None:
    # Very simple parser for the specific format requested:
    # objects:{id:N,]{headers}:
    # row1
    # row2
    # objects:{}
    
    text = text.strip()
    
    # Check for empty objects
    if "objects:{}" in text:
        return {"objects": []}

    # Regex for table header
    # "objects:{id:2,]{class_name,x,y,width,height}:"
    header_pattern = r"objects:\{id:\d+,\]\{([^}]+)\}:"
    match = re.search(header_pattern, text)
    
    if not match:
        return None
    
    headers = [h.strip() for h in match.group(1).split(",")]
    
    # Extract rows (lines after header match)
    start_pos = match.end()
    lines = text[start_pos:].strip().splitlines()
    
    objects = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("objects:"): # End marker or new block
            continue
            
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= len(headers):
            obj = {}
            for i, h in enumerate(headers):
                val = parts[i]
                # Convert to int/float if possible
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                obj[h] = val
            objects.append(obj)
            
    return {"objects": objects}

def toon_available() -> bool:
    return True

def wrap_prompt(instruction: str, schema: Any = None) -> str:
    return instruction

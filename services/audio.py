"""
services/audio.py
Launcher for the Effector Audio Service.
"""
import sys
from pathlib import Path

# Ensure the package is in path if not installed in editable mode
sys.path.insert(0, str(Path(__file__).parent.parent))

from effector.audio.service import mcp, _startup

if __name__ == "__main__":
    _startup()
    # Run the FastMCP server on port 5558 as dictated by the roadmap
    # FastMCP uses SSE/HTTP by default when run via .run() 
    # (Check FastMCP docs for exact host/port kwargs depending on version)
    mcp.run()
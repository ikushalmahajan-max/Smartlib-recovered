"""
Streamlit Cloud Entry Point wrapper.
Imports and runs the main app.py application.
"""

import runpy
import os
import sys

# Add directory to sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run main app script
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), run_name="__main__")

"""
Virtual Laboratory Experiment: Construction of an Inverted Index
Course: Knowledge Graphs and Information Retrieval Systems (KGIRS)
Experiment No: 3 (Allocated Roll Numbers: 11 - 15)
Institute: Indian Institute of Technology Kharagpur (IIT Kharagpur) - Virtual Labs CA

Wrapper module enabling seamless execution via either:
    streamlit run template.py
or
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure directory is on sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from app import main

if __name__ == "__main__":
    main()

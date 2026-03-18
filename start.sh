#!/bin/bash
echo
echo "  Starting BrainBox..."
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "  Python 3 not found! Install it from https://www.python.org/downloads/"
    exit 1
fi

# Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "  Installing dependencies (first time only)..."
    pip3 install -r requirements.txt -q
    touch .deps_installed
    echo "  Done!"
    echo
fi

# Start
python3 app.py

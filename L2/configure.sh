#!/bin/bash

echo "Setting up Python virtual environment for exaSIMPLE L2 scripts ..."

# Exit on any error
set -e

# Default values
ENV_NAME="venv"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env_name)
            ENV_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./configure.sh [--env_name NAME]"
            echo "Options:"
            echo "  --env_name NAME   Name of the virtual environment directory (default: venv)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed or not in your PATH."
    echo " Please install Python 3.8+ before running this script."
    exit 1
fi

# Create and activate virtual environment
python3 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages
echo "Installing required packages..."
pip install numpy scipy pandas filelock

echo "Python environment setup complete!"
echo "To activate later, run:"
echo "source $ENV_NAME/bin/activate"

echo ""
echo "PETSc must be installed separately with MPI support."
echo "Set PETSC_DIR and PETSC_ARCH before running the solver:"
echo "export PETSC_DIR=/path/to/petsc"
echo "export PETSC_ARCH=arch-linux-c-opt"

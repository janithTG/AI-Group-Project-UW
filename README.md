# AI-Group-Project-UW

MSc AI and Automation Group Project

## Setup

### Prerequisites

- Python 3.10+ installed ([python.org](https://www.python.org/downloads/))
- On Windows, tick **Add Python to PATH** during install

### 1. Clone the repository

```bash
git clone https://github.com/janithTG/AI-Group-Project-UW.git
cd AI-Group-Project-UW
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
```

**Windows (Command Prompt or PowerShell)**

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt)**

```bat
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once (as Administrator if needed):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

When the venv is active, your prompt usually starts with `(.venv)`.

### 4. Install packages

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Jupyter notebook kernel (optional)

Register the project kernel so notebooks use this environment:

```bash
python -m ipykernel install --user --name=ai-group-project --display-name="Python (AI Group Project)"
```

Then open `notebooks/dataset_explore.ipynb` and select **Python (AI Group Project)** as the kernel.

### 6. Deactivate when finished

```bash
deactivate
```

## Project structure

```
data/
  raw/           # original datasets
  processed/     # cleaned / transformed data
docs/            # documentation
notebooks/       # exploratory notebooks
src/             # source code
tests/           # tests
requirements.txt # Python dependencies
```

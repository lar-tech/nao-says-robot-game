# nao-says-robot-game

## Setup
Working with the NAO robot on macOS is challenging because the NAOqi SDK only supports Linux. To solve this, we use a Docker container with the `pynaoqi-python2.7-2.1.2.17-linux64.tar` SDK.

The project uses two Python environments:
- **Python 3.13**: Runs the ML models (Whisper, etc.)
- **Python 2.7**: Handles robot communication via NAOqi SDK (runs as subprocess inside Docker)

The project requires the following structure (note: Docker files are placed *outside* the repository):
```
parent_directory/
├── Dockerfile
├── run-naoqi.sh
├── pynaoqi-python2.7-2.1.2.17-linux64.tar
└── nao-says-robot-game/          # this repository
    └── ...
```

### Installation

1. Build the Docker image:
```bash
   docker build --platform=linux/amd64 -t naoqi .
```

2. Make the shell script executable:
```bash
   chmod +x run-naoqi.sh
```

3. Install the Python package:
```bash
   cd nao-says-robot-game
   pip install -e .
```

4. Run the application:
```bash
   python src/nao_says/execute.py
```

   On first run, all required models (Whisper, etc.) will be downloaded automatically.
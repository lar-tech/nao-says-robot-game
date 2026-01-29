# NAO Says Robot Game

An interactive voice-controlled "Simon Says" game for the NAO humanoid robot. Users speak commands starting with "Simon Says", the robot executes actions (postures, movements, speech), captures images, performs object detection, and describes what it sees.

## Project Structure

```
naoqi-docker/                         # Parent directory
├── Dockerfile                        # Docker image for Python 2.7 + NAOqi
├── run-naoqi.sh                      # Shell script to run Docker container
├── pynaoqi-python2.7-2.1.2.17-linux64.tar  # NAOqi SDK
└── nao-says-robot-game/              # This repository
    ├── src/
    │   ├── nao_says/                 # Python 3 application (ML models)
    │   │   ├── execute.py            # Main entry point
    │   │   ├── voice.py              # Voice command processing (Whisper)
    │   │   └── vision.py             # Object detection (YOLOv8)
    │   └── nao_bundle/               # Python 2.7 application (Robot control)
    │       ├── execute.py            # NAOqi command dispatcher
    │       ├── tasks.py              # Robot task executor
    │       └── inao/                 # Enhanced NAOqi proxy wrapper
    ├── pyproject.toml                # Package configuration
    └── README.md
```

## Pipeline

```

                    PYTHON 3.13 (Host Machine)                   
                                                                 
  1. Voice Input                                                 
     └─→ Record 5s audio                                         
     └─→ Whisper transcription                                   
     └─→ Detect "simon says" wake word                           
     └─→ Map voice to command JSON                               
                                                                 
  2. Send command to Docker subprocess via stdin                 

               DOCKER CONTAINER (Python 2.7 + NAOqi)             
                                                                 
  3. Command Dispatch                                            
     └─→ Parse JSON command                                      
     └─→ Execute robot action:                                   
         • Postures (Stand, Sit, Crouch, etc.)                   
         • Movement (Forward, Backward, Turn)                    
         • Joint control (Arms, Head)                            
         • Eye color (LEDs)                                      
         • Text-to-speech                                        
     └─→ Capture image from robot camera                         
     └─→ Return Base64 JPEG via stdout                           

                    PYTHON 3.13 (Host Machine)                   
                                                                 
  4. Vision Processing                                           
     └─→ Decode Base64 image                                     
     └─→ Run YOLOv8 object detection                             
     └─→ Filter targets (person, bottle, ball, etc.)             
                                                                 
  5. Response                                                    
     └─→ Generate description: "I see 1 person, 2 bottles"       
     └─→ Send say_text command to robot                          
     └─→ Display annotated image                                 
                                                                 
  6. Loop until "game over"                       

```

## Technologies

| Component | Technology |
|-----------|------------|
| Speech Recognition | Faster Whisper (Small) |
| Object Detection | YOLOv8n |
| Robot Control | NAOqi SDK (Python 2.7) |
| Image Processing | OpenCV |
| Containerization | Docker (Ubuntu 20.04, linux/amd64) |

## Supported Commands

| Category | Commands |
|----------|----------|
| Postures | Stand, Stand Zero, Stand Init, Crouch, Sit, Sit Relax, Lying Belly, Lying Back |
| Movement | Forward, Backward, Left, Right, Turn Left, Turn Right |
| Head/Arms | Rotate Head, Move Head, Lift Arms, Stretch Elbows, Bend Elbows, Twist Wrists |
| Other | Change Eye Color, Capture Photo, Say Text, Game Over |

## Setup

Working with the NAO robot on macOS is challenging because the NAOqi SDK only supports Linux. To solve this, we use a Docker container with the `pynaoqi-python2.7-2.1.2.17-linux64.tar` SDK.

The project uses two Python environments:
- **Python 3.13**: Runs the ML models (Whisper, YOLOv8)
- **Python 2.7**: Handles robot communication via NAOqi SDK (runs as subprocess inside Docker)

### Installation

1. Build the Docker image (from parent directory):
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

4. Configure the robot IP in `src/nao_says/execute.py`

5. Run the application:
```bash
python src/nao_says/execute.py
```

On first run, all required models (Whisper, YOLOv8) will be downloaded automatically.

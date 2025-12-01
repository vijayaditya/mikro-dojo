# MIKRO-DOJO Modular Architecture Design

## Overview

This document defines the modular, extensible architecture for MIKRO-DOJO, built on proven open-source foundations. The design prioritizes:

- **Modularity**: Loosely coupled components with clean interfaces
- **Extensibility**: Easy addition of new cars, arenas, skills, and integrations
- **Open Source First**: Leverage battle-tested projects over custom solutions
- **Future-Ready**: Architecture supports Scratch/kids tools integration

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MIKRO-DOJO PLATFORM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   SCRATCH   │  │   WEB UI    │  │  REST API   │  │   PYTHON SDK        │ │
│  │  EXTENSION  │  │  (React)    │  │  (FastAPI)  │  │   (mikro-dojo-py)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                     │           │
│  ═══════╪════════════════╪════════════════╪═════════════════════╪═════════  │
│         │         INTEGRATION GATEWAY (WebSocket + REST)        │           │
│  ═══════╪════════════════════════════════════════════════════════╪═════════  │
│         │                                                        │           │
│  ┌──────┴────────────────────────────────────────────────────────┴────────┐ │
│  │                        ORCHESTRATION LAYER                              │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐│ │
│  │  │ Skill      │  │ Session    │  │ Fleet      │  │ Curriculum         ││ │
│  │  │ Manager    │  │ Controller │  │ Manager    │  │ Engine             ││ │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐  │
│  │                         ROS 2 MESSAGE BUS                              │  │
│  │                    (Iron Irwini / Jazzy Jalisco)                       │  │
│  └─────────────────────────────────┬─────────────────────────────────────┘  │
│                                    │                                         │
│  ┌────────────┬────────────┬───────┴───────┬────────────┬────────────────┐  │
│  │            │            │               │            │                │  │
│  ▼            ▼            ▼               ▼            ▼                ▼  │
│ ┌────────┐ ┌────────┐ ┌─────────┐ ┌────────────┐ ┌──────────┐ ┌─────────┐  │
│ │PERCEP- │ │LEARNING│ │EVALUA-  │ │ EXECUTION  │ │  DATA    │ │ SAFETY  │  │
│ │TION    │ │ENGINE  │ │TION     │ │ RUNTIME    │ │ PIPELINE │ │ SYSTEM  │  │
│ └────────┘ └────────┘ └─────────┘ └────────────┘ └──────────┘ └─────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           HARDWARE ABSTRACTION                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  ARENA DRIVERS  │  │  VEHICLE DRIVERS │  │  SENSOR DRIVERS            │  │
│  │  (Multi-Arena)  │  │  (Multi-Car)     │  │  (Cameras, IMU, Encoders)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Open-Source Foundations

### 1. Robot Framework: ROS 2

**Why ROS 2?**
- Industry standard for robotics
- Built-in support for multi-robot systems
- Hardware abstraction layer
- Extensive package ecosystem
- Native Python and C++ support

**Distribution**: ROS 2 Jazzy Jalisco (Ubuntu 24.04) or Iron Irwini (Ubuntu 22.04)

```yaml
ros2_packages:
  core:
    - ros-jazzy-ros-base
    - ros-jazzy-rclpy
    - ros-jazzy-std-msgs
    - ros-jazzy-sensor-msgs
    - ros-jazzy-geometry-msgs

  perception:
    - ros-jazzy-image-transport
    - ros-jazzy-cv-bridge
    - ros-jazzy-camera-calibration
    - ros-jazzy-image-pipeline

  navigation:
    - ros-jazzy-tf2-ros
    - ros-jazzy-robot-localization

  visualization:
    - ros-jazzy-rviz2
    - ros-jazzy-foxglove-bridge  # Web-based visualization
```

### 2. Robot Learning: LeRobot (Hugging Face)

**Why LeRobot?**
- Purpose-built for robot learning from demonstrations
- Supports imitation learning and RL
- Pre-built dataset formats
- Model hub integration
- Active development by Hugging Face

```yaml
lerobot_components:
  - lerobot.common.datasets     # Dataset management
  - lerobot.common.policies     # Policy implementations
  - lerobot.common.robot_devices # Hardware interfaces
  - lerobot.scripts.train       # Training pipelines
```

**Integration Points**:
- Use LeRobot's dataset format for demonstrations
- Extend LeRobot policies for MIKRO-DOJO skills
- Leverage their replay buffer implementations

### 3. Vision & Perception

```yaml
perception_stack:
  core_vision:
    - opencv-python          # Core CV operations
    - opencv-contrib-python  # Extended features

  camera_calibration:
    - kalibr                 # Multi-camera calibration (ETH Zurich)
    - pupil-apriltags        # Fiducial detection

  3d_reconstruction:
    - open3d                 # Point cloud processing
    - pycolmap               # Structure from motion (optional)

  video_processing:
    - ffmpeg-python          # Video encoding/decoding
    - decord                  # Fast video loading for ML
```

### 4. Machine Learning Stack

```yaml
ml_framework:
  core:
    - pytorch >= 2.0         # Primary ML framework
    - torchvision            # Vision models
    - torchaudio             # Audio (future)

  reinforcement_learning:
    - stable-baselines3      # RL algorithms (PPO, SAC)
    - gymnasium              # Environment interface
    - tianshou               # Alternative RL library

  imitation_learning:
    - imitation              # IL algorithms
    - d3rlpy                 # Offline RL

  transformers:
    - transformers           # Hugging Face models
    - accelerate             # Distributed training
```

### 5. Vision-Language Models (VLM)

```yaml
vlm_options:
  open_source:
    - llava-next             # LLaVA 1.6+ (Apache 2.0)
    - qwen-vl                # Qwen-VL (open weights)
    - cogvlm                 # CogVLM (Apache 2.0)
    - moondream              # Lightweight VLM

  integration:
    - vllm                   # Fast inference server
    - ollama                 # Local model serving
    - litellm                # Unified API wrapper
```

### 6. Edge Deployment

```yaml
edge_stack:
  nvidia_jetson:
    - jetpack >= 6.0         # JetPack SDK
    - tensorrt               # Model optimization
    - isaac-ros              # NVIDIA ROS packages
    - deepstream             # Video analytics

  model_optimization:
    - onnx                   # Model format
    - onnxruntime            # Cross-platform inference
    - torch-tensorrt         # PyTorch → TensorRT
```

### 7. Data Pipeline

```yaml
data_stack:
  storage:
    - apache-arrow           # Columnar data format
    - pyarrow                # Python bindings
    - lance                  # ML-native format (optional)

  versioning:
    - dvc                    # Data version control
    - git-lfs                # Large file storage

  experiment_tracking:
    - mlflow                 # Experiment tracking
    - wandb                  # Weights & Biases (optional)

  databases:
    - sqlite                 # Local metadata
    - redis                  # Caching/queues
```

### 8. Communication

```yaml
communication:
  internal:
    - ros2-dds               # ROS 2 DDS (default: FastDDS)
    - grpcio                 # High-performance RPC

  external:
    - fastapi                # REST API
    - websockets             # Real-time web
    - paho-mqtt              # IoT messaging

  serialization:
    - protobuf               # Binary serialization
    - msgpack                # Compact JSON alternative
```

---

## Module Specifications

### Module 1: Hardware Abstraction Layer (HAL)

**Purpose**: Abstract all hardware into consistent interfaces

```
mikro_dojo/
└── hal/
    ├── __init__.py
    ├── interfaces/
    │   ├── vehicle.py       # IVehicle abstract base
    │   ├── arena.py         # IArena abstract base
    │   ├── camera.py        # ICamera abstract base
    │   └── sensor.py        # ISensor abstract base
    │
    ├── vehicles/
    │   ├── base_rc_car.py   # Base RC car implementation
    │   ├── traxxas_slash.py # Traxxas Slash 4x4
    │   ├── arrma_granite.py # Arrma Granite
    │   ├── wltoys_124019.py # Budget option
    │   └── mikro_carrier.py # Large carrier vehicle with plane storage
    │
    ├── arenas/
    │   ├── base_arena.py    # Base arena implementation
    │   ├── home_arena.py    # Small home setup (3m x 3m)
    │   └── lab_arena.py     # Full lab setup (5m x 5m)
    │
    └── sensors/
        ├── cameras/
        │   ├── usb_camera.py
        │   ├── csi_camera.py     # Jetson CSI
        │   └── ip_camera.py      # Network cameras
        ├── imu/
        │   ├── bno055.py
        │   └── icm20948.py
        └── encoders/
            └── quadrature.py
```

**Vehicle Interface**:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class VehicleState:
    position: tuple[float, float, float]  # x, y, theta
    velocity: tuple[float, float, float]  # vx, vy, omega
    battery_voltage: float
    motor_currents: list[float]

class IVehicle(ABC):
    """Abstract interface for all vehicle types"""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def send_command(self, throttle: float, steering: float) -> None: ...

    @abstractmethod
    def get_state(self) -> VehicleState: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @property
    @abstractmethod
    def vehicle_id(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> dict: ...
```

**Large Vehicle: MIKRO-CARRIER**

For missions requiring high plane capacity and extended operations, the MIKRO-CARRIER is a larger vehicle platform:

```python
@VehicleRegistry.register("mikro_carrier")
class MikroCarrier(IVehicle):
    """Large carrier vehicle with integrated plane storage and launcher"""

    SPECIFICATIONS = {
        "scale": "1/5",                    # Larger than standard 1/10
        "length_cm": 80,                   # 80cm long
        "width_cm": 45,                    # 45cm wide
        "height_cm": 35,                   # 35cm tall (with launcher)
        "weight_kg": 8.0,                  # 8kg base weight
        "payload_kg": 5.0,                 # 5kg payload capacity
        "max_speed_kmh": 40,               # 40 km/h max speed
        "drive": "6WD",                    # 6-wheel drive for stability
        "battery": "6S 10000mAh LiPo",     # Large battery
        "runtime_minutes": 45,             # 45 min runtime
    }

    FEATURES = [
        "plane_storage_bay",       # Rear cargo bay for 10-20 planes
        "roof_launcher",           # Electromagnetic catapult launcher
        "front_camera",            # Wide-angle stereo camera
        "voice_control",           # Dictation control system
        "gps_navigation",          # Outdoor GPS support
        "4g_connectivity",         # Remote operation capability
    ]
```

**MIKRO-CARRIER Specifications**:
| Specification | Value |
|---------------|-------|
| Scale | 1/5 (larger than standard) |
| Dimensions | 80cm × 45cm × 35cm |
| Base Weight | 8 kg |
| Payload Capacity | 5 kg |
| Drive System | 6WD independent motors |
| Top Speed | 40 km/h |
| Battery | 6S 10000mAh LiPo |
| Runtime | 45 minutes |
| Plane Capacity | 10-20 planes |
| Compute | Jetson AGX Orin |

**Why Bigger?**
- More space for plane storage magazine (10-20 planes)
- Larger battery for extended missions
- More powerful motors to carry payload
- Room for advanced compute (Jetson AGX Orin)
- Better stability for launching planes while moving
- Space for voice control microphone array

### Module 1.5: Vehicle Accessories

**Plane Launcher Module**:
```
mikro_dojo/
└── hal/
    └── accessories/
        ├── __init__.py
        ├── plane_launcher.py      # Plane launcher interface & implementation
        ├── plane_storage.py       # Plane storage bay management
        ├── front_camera.py        # Front camera module
        └── voice_control.py       # Dictation/voice control system
```

**Plane Launcher Interface**:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class LauncherState(Enum):
    IDLE = "idle"
    ARMED = "armed"
    LAUNCHING = "launching"
    RELOADING = "reloading"

@dataclass
class PlaneConfig:
    plane_type: str           # "glider", "powered", "foam"
    weight_grams: float       # Plane weight
    wingspan_cm: float        # Wingspan for flight planning

@dataclass
class LaunchParameters:
    angle_degrees: float      # Launch angle (0-45)
    power_percent: float      # Launch power (0-100)
    delay_ms: int             # Delay before launch

class IPlaneLauncher(ABC):
    """Interface for roof-mounted plane launcher"""

    @abstractmethod
    def arm(self) -> bool:
        """Prepare launcher for firing"""
        ...

    @abstractmethod
    def launch(self, params: LaunchParameters) -> bool:
        """Launch the plane with specified parameters"""
        ...

    @abstractmethod
    def get_state(self) -> LauncherState:
        """Get current launcher state"""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if plane is loaded"""
        ...

    @abstractmethod
    def get_planes_remaining(self) -> int:
        """Get number of planes in magazine"""
        ...

    @property
    @abstractmethod
    def max_capacity(self) -> int:
        """Maximum planes the launcher can hold"""
        ...
```

**Launcher Hardware Specifications**:
- **Mounting**: Roof-mounted rail system with quick-release
- **Capacity**: 10-20 planes (fed from storage bay)
- **Launch Mechanism**: Electromagnetic catapult or compressed air cannon
- **Launch Angle**: Adjustable 0-60 degrees via high-torque servo
- **Power Source**: Dedicated high-capacity LiPo (4S-6S)
- **Fire Rate**: Up to 2 planes per second in rapid mode
- **Trigger**: Electronic servo-actuated release with voice activation support

**Supported Plane Types**:
| Type | Weight | Wingspan | Flight Time | Use Case |
|------|--------|----------|-------------|----------|
| Foam Glider | 20-50g | 30-50cm | 10-30s glide | Basic aerial observation |
| Powered Mini | 50-100g | 40-60cm | 2-5 min | Extended reconnaissance |
| FPV Micro | 80-150g | 50-80cm | 3-8 min | First-person view missions |

**Plane Storage Bay**:
```python
@dataclass
class StorageBayConfig:
    capacity: int             # 10-20 planes
    plane_type: str           # Type of planes stored
    auto_reload: bool         # Automatic feeding to launcher
    climate_controlled: bool  # Temperature regulation for batteries

class IPlaneStorage(ABC):
    """Interface for plane storage management"""

    @abstractmethod
    def get_inventory(self) -> list[PlaneConfig]:
        """Get list of stored planes"""
        ...

    @abstractmethod
    def feed_to_launcher(self) -> bool:
        """Transfer plane from storage to launcher"""
        ...

    @abstractmethod
    def get_capacity(self) -> tuple[int, int]:
        """Return (current_count, max_capacity)"""
        ...
```

**Storage Hardware Specifications**:
- **Location**: Rear cargo bay with conveyor feed system
- **Capacity**: 10-20 planes in stacked magazine configuration
- **Auto-Feed**: Motorized conveyor belt to launcher
- **Plane Protection**: Foam-lined compartments prevent damage
- **Status Indicators**: LED display showing plane count
- **Quick Reload**: Slide-out magazine for fast reloading

**Front Camera Module**:
The vehicle includes a forward-facing camera system for:
- Obstacle detection and avoidance
- Visual navigation and SLAM
- Recording skill execution from ego perspective
- Real-time streaming for teleoperation

**Camera Specifications**:
- **Type**: Wide-angle stereo or monocular
- **Resolution**: 720p-1080p at 30-60fps
- **Field of View**: 90-120 degrees horizontal
- **Mounting**: Hood-mounted or bumper-integrated
- **Features**: Low-light capable, global shutter preferred

**Voice Control / Dictation System**:
Control the car using your voice! Speak commands naturally and the car responds.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class VoiceCommandType(Enum):
    MOVEMENT = "movement"       # "go forward", "turn left", "stop"
    SPEED = "speed"             # "go faster", "slow down", "full speed"
    LAUNCHER = "launcher"       # "launch plane", "arm launcher", "fire"
    SKILL = "skill"             # "do a drift", "parallel park", "donut"
    QUERY = "query"             # "how many planes left?", "battery status?"

@dataclass
class VoiceCommand:
    raw_text: str               # Original spoken text
    command_type: VoiceCommandType
    action: str                 # Parsed action
    parameters: dict            # Extracted parameters
    confidence: float           # Recognition confidence (0-1)

class IVoiceControl(ABC):
    """Interface for dictation/voice control"""

    @abstractmethod
    def start_listening(self) -> None:
        """Begin listening for voice commands"""
        ...

    @abstractmethod
    def stop_listening(self) -> None:
        """Stop listening"""
        ...

    @abstractmethod
    def process_audio(self, audio_data: bytes) -> VoiceCommand:
        """Process audio and return parsed command"""
        ...

    @abstractmethod
    def get_supported_commands(self) -> list[str]:
        """Return list of supported voice commands"""
        ...

    @abstractmethod
    def set_wake_word(self, word: str) -> None:
        """Set custom wake word (e.g., 'Hey Car', 'OK Racer')"""
        ...
```

**Voice Control Hardware**:
- **Microphone**: Directional MEMS microphone array (2-4 mics)
- **Processing**: On-device speech recognition (Whisper/Vosk)
- **Wake Word**: Customizable ("Hey Car", "OK Racer", etc.)
- **Noise Cancellation**: Active noise filtering for outdoor use
- **Response**: Speaker for audio feedback and confirmations

**Supported Voice Commands**:
| Category | Example Commands |
|----------|------------------|
| Movement | "go forward", "turn left", "turn right", "reverse", "stop" |
| Speed | "go faster", "slow down", "half speed", "full speed" |
| Launcher | "arm launcher", "launch plane", "fire", "launch all" |
| Skills | "do a drift", "do a donut", "parallel park", "slalom" |
| Status | "battery level", "how many planes", "speed check" |

**Voice Control Integration**:
```yaml
voice_control:
  enabled: true
  wake_word: "hey car"
  language: "en-US"
  sensitivity: 0.7
  continuous_listening: false
  confirmation_audio: true
  commands:
    movement:
      forward: ["go forward", "drive", "move ahead", "go"]
      backward: ["reverse", "go back", "back up"]
      left: ["turn left", "go left", "left"]
      right: ["turn right", "go right", "right"]
      stop: ["stop", "halt", "freeze", "brake"]
    launcher:
      arm: ["arm launcher", "prepare to launch", "get ready"]
      fire: ["launch", "fire", "launch plane", "send it"]
      status: ["planes remaining", "how many planes", "ammo check"]
```

### Module 2: Perception System

**Purpose**: Multi-view observation, synchronization, and processing

```
mikro_dojo/
└── perception/
    ├── __init__.py
    ├── capture/
    │   ├── multi_camera.py      # Synchronized capture
    │   ├── sync_manager.py      # Hardware/software sync
    │   └── stream_recorder.py   # Recording to disk
    │
    ├── calibration/
    │   ├── intrinsic.py         # Single camera calibration
    │   ├── extrinsic.py         # Multi-camera calibration
    │   ├── arena_mapping.py     # World coordinate system
    │   └── kalibr_wrapper.py    # Kalibr integration
    │
    ├── tracking/
    │   ├── vehicle_tracker.py   # Multi-vehicle tracking
    │   ├── apriltag_detector.py # Fiducial markers
    │   └── visual_odometry.py   # VO pipeline
    │
    └── correspondence/
        ├── ego_exo_matcher.py   # Ego-exo view matching
        ├── view_synthesis.py    # Novel view generation
        └── embeddings.py        # Visual embeddings
```

**ROS 2 Topics**:
```yaml
perception_topics:
  inputs:
    - /arena/camera_{n}/image_raw      # Raw images
    - /arena/camera_{n}/camera_info    # Camera intrinsics
    - /vehicle_{id}/camera/image_raw   # Onboard camera
    - /vehicle_{id}/imu/data           # IMU readings

  outputs:
    - /perception/sync_frames          # Synchronized multi-view
    - /perception/vehicle_poses        # Tracked vehicle poses
    - /perception/visual_embeddings    # Learned embeddings
```

### Module 3: Learning Engine

**Purpose**: Skill acquisition through demonstration and reinforcement

```
mikro_dojo/
└── learning/
    ├── __init__.py
    ├── datasets/
    │   ├── mikro_dataset.py     # LeRobot-compatible dataset
    │   ├── demonstration.py     # Demo recording/loading
    │   └── augmentation.py      # Data augmentation
    │
    ├── primitives/
    │   ├── extractor.py         # Behavioral primitive extraction
    │   ├── library.py           # Primitive library
    │   └── composer.py          # Primitive composition
    │
    ├── policies/
    │   ├── base_policy.py       # Policy interface
    │   ├── diffusion_policy.py  # Diffusion-based (LeRobot)
    │   ├── act_policy.py        # Action Chunking Transformer
    │   └── bc_policy.py         # Behavioral cloning
    │
    ├── reinforcement/
    │   ├── gym_env.py           # Gymnasium environment
    │   ├── reward_functions.py  # Reward engineering
    │   ├── ppo_trainer.py       # PPO training loop
    │   └── sac_trainer.py       # SAC training loop
    │
    └── training/
        ├── trainer.py           # Unified training interface
        ├── callbacks.py         # Training callbacks
        └── distributed.py       # Multi-GPU training
```

**LeRobot Integration**:
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

class MikroDojoDataset(LeRobotDataset):
    """MIKRO-DOJO dataset extending LeRobot format"""

    def __init__(self, repo_id: str, **kwargs):
        super().__init__(repo_id, **kwargs)
        # Add multi-view support
        self.exo_cameras = ["arena_cam_0", "arena_cam_1", "arena_cam_2", "arena_cam_3"]
        self.ego_cameras = ["vehicle_front", "vehicle_rear"]

class MikroDojoPolicy(DiffusionPolicy):
    """Extended policy with multi-view input"""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.ego_exo_encoder = EgoExoEncoder()  # Custom encoder
```

### Module 4: Evaluation System

**Purpose**: VLM-guided assessment and skill evaluation

```
mikro_dojo/
└── evaluation/
    ├── __init__.py
    ├── vlm/
    │   ├── base_evaluator.py    # VLM interface
    │   ├── llava_evaluator.py   # LLaVA implementation
    │   ├── qwen_evaluator.py    # Qwen-VL implementation
    │   ├── prompts.py           # Evaluation prompts
    │   └── ollama_client.py     # Local serving
    │
    ├── metrics/
    │   ├── trajectory.py        # Trajectory similarity
    │   ├── success_rate.py      # Task completion
    │   ├── safety.py            # Safety violations
    │   └── efficiency.py        # Time/energy metrics
    │
    └── visualization/
        ├── replay.py            # Episode replay
        ├── comparison.py        # Side-by-side comparison
        └── dashboard.py         # Metrics dashboard
```

**VLM Evaluator Interface**:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    task_completed: bool
    quality_score: float  # 0-10
    safety_violations: list[str]
    style_score: float
    feedback: str

class IVLMEvaluator(ABC):
    """Interface for VLM-based evaluation"""

    @abstractmethod
    def evaluate_episode(
        self,
        frames: list[np.ndarray],
        skill_name: str,
        success_criteria: str
    ) -> EvaluationResult: ...

    @abstractmethod
    def compare_attempts(
        self,
        attempt_a: list[np.ndarray],
        attempt_b: list[np.ndarray],
        skill_name: str
    ) -> tuple[int, str]: ...  # winner, explanation
```

### Module 5: Execution Runtime

**Purpose**: Real-time policy execution with safety guarantees

```
mikro_dojo/
└── execution/
    ├── __init__.py
    ├── runtime/
    │   ├── policy_executor.py   # Policy inference loop
    │   ├── action_smoother.py   # Action smoothing
    │   └── state_estimator.py   # State fusion
    │
    ├── safety/
    │   ├── safety_monitor.py    # Real-time monitoring
    │   ├── constraints.py       # Action constraints
    │   ├── collision_detect.py  # Collision prediction
    │   └── emergency_stop.py    # E-stop system
    │
    └── deployment/
        ├── model_loader.py      # ONNX/TensorRT loading
        ├── optimizer.py         # Model optimization
        └── edge_runtime.py      # Jetson-specific runtime
```

### Module 6: Integration Gateway

**Purpose**: External interfaces for APIs, UIs, and future tools (Scratch)

```
mikro_dojo/
└── gateway/
    ├── __init__.py
    ├── api/
    │   ├── rest_api.py          # FastAPI REST endpoints
    │   ├── websocket.py         # Real-time WebSocket
    │   └── graphql.py           # GraphQL (optional)
    │
    ├── sdk/
    │   ├── python_sdk.py        # Python client library
    │   └── typescript_sdk/      # TypeScript client
    │
    └── integrations/
        ├── scratch/
        │   ├── extension.js     # Scratch 3.0 extension
        │   ├── blocks.py        # Block definitions
        │   └── bridge.py        # Python ↔ Scratch bridge
        │
        ├── blockly/
        │   ├── toolbox.js       # Blockly toolbox
        │   └── generators.js    # Code generators
        │
        └── web_ui/
            └── (React app)
```

**Scratch Integration Design**:
```javascript
// scratch_extension.js
class MikroDojoExtension {
    getInfo() {
        return {
            id: 'mikroDojo',
            name: 'MIKRO-DOJO',
            blocks: [
                {
                    opcode: 'driveForward',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'drive [CAR] forward at [SPEED]%',
                    arguments: {
                        CAR: { type: Scratch.ArgumentType.STRING, defaultValue: 'car1' },
                        SPEED: { type: Scratch.ArgumentType.NUMBER, defaultValue: 50 }
                    }
                },
                {
                    opcode: 'performSkill',
                    blockType: Scratch.BlockType.COMMAND,
                    text: '[CAR] do [SKILL]',
                    arguments: {
                        CAR: { type: Scratch.ArgumentType.STRING, defaultValue: 'car1' },
                        SKILL: {
                            type: Scratch.ArgumentType.STRING,
                            menu: 'skillMenu',
                            defaultValue: 'drift'
                        }
                    }
                }
            ],
            menus: {
                skillMenu: ['drift', 'donut', 'parallel_park', 'slalom']
            }
        };
    }
}
```

---

## Plugin Architecture

For maximum extensibility, each major component supports plugins:

```python
# mikro_dojo/core/plugins.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')

class PluginRegistry(Generic[T]):
    """Generic plugin registry for extensibility"""

    _plugins: dict[str, type[T]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(plugin_class: type[T]) -> type[T]:
            cls._plugins[name] = plugin_class
            return plugin_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type[T]:
        return cls._plugins[name]

    @classmethod
    def list_plugins(cls) -> list[str]:
        return list(cls._plugins.keys())

# Usage example for vehicles
class VehicleRegistry(PluginRegistry[IVehicle]):
    pass

@VehicleRegistry.register("traxxas_slash")
class TraxxasSlash(IVehicle):
    ...

@VehicleRegistry.register("arrma_granite")
class ArrmaGranite(IVehicle):
    ...

@VehicleRegistry.register("mikro_carrier")
class MikroCarrier(IVehicle):
    """Large 1/5 scale carrier with plane storage and voice control"""
    ...
```

---

## Multi-Car / Multi-Arena Support

### Fleet Management

```python
# mikro_dojo/orchestration/fleet.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class VehicleConfig:
    vehicle_id: str
    vehicle_type: str  # Plugin name
    arena_id: str
    capabilities: dict

@dataclass
class ArenaConfig:
    arena_id: str
    arena_type: str  # Plugin name
    cameras: list[str]
    bounds: tuple[float, float, float]  # width, depth, height

class FleetManager:
    """Manages multiple vehicles across multiple arenas"""

    def __init__(self):
        self.vehicles: dict[str, IVehicle] = {}
        self.arenas: dict[str, IArena] = {}

    def register_vehicle(self, config: VehicleConfig) -> IVehicle:
        vehicle_class = VehicleRegistry.get(config.vehicle_type)
        vehicle = vehicle_class(config)
        self.vehicles[config.vehicle_id] = vehicle
        return vehicle

    def register_arena(self, config: ArenaConfig) -> IArena:
        arena_class = ArenaRegistry.get(config.arena_type)
        arena = arena_class(config)
        self.arenas[config.arena_id] = arena
        return arena

    def get_vehicles_in_arena(self, arena_id: str) -> list[IVehicle]:
        return [v for v in self.vehicles.values()
                if v.arena_id == arena_id]
```

### Configuration-Driven Setup

```yaml
# config/fleet.yaml
arenas:
  - id: home_arena
    type: home_arena
    cameras:
      - id: overhead_1
        type: usb_camera
        device: /dev/video0
        position: [1.5, 1.5, 2.0]
      - id: corner_1
        type: ip_camera
        url: rtsp://192.168.1.100/stream
        position: [0, 0, 1.5]
    bounds: [3.0, 3.0, 2.0]
    floor_friction: 0.8

vehicles:
  - id: car_alpha
    type: traxxas_slash
    arena: home_arena
    connection:
      type: bluetooth
      address: "AA:BB:CC:DD:EE:FF"
    sensors:
      imu: bno055
      camera: csi_camera
    accessories:
      front_camera:
        enabled: true
        type: wide_angle_stereo
        resolution: 720p
        fps: 30
      plane_launcher:
        enabled: true
        type: spring_loaded
        capacity: 2
        plane_type: foam_glider

  - id: car_beta
    type: wltoys_124019
    arena: home_arena
    connection:
      type: wifi
      ip: 192.168.1.101
    accessories:
      front_camera:
        enabled: true
        type: monocular
        resolution: 1080p
        fps: 60

  # Large carrier vehicle with full plane launching capabilities
  - id: carrier_one
    type: mikro_carrier
    arena: home_arena
    connection:
      type: wifi
      ip: 192.168.1.102
    sensors:
      imu: bno085
      camera: stereo_csi
      gps: ublox_m9n
    accessories:
      front_camera:
        enabled: true
        type: wide_angle_stereo
        resolution: 1080p
        fps: 60
      plane_launcher:
        enabled: true
        type: electromagnetic_catapult
        capacity: 20
        plane_type: foam_glider
        auto_reload: true
      plane_storage:
        enabled: true
        capacity: 20
        auto_feed: true
        magazine_type: stacked
      voice_control:
        enabled: true
        wake_word: "hey carrier"
        language: "en-US"
        continuous_listening: true
```

---

## ROS 2 Package Structure

```
mikro_dojo_ws/
└── src/
    ├── mikro_dojo_msgs/         # Custom message definitions
    │   ├── msg/
    │   │   ├── VehicleState.msg
    │   │   ├── VehicleCommand.msg
    │   │   ├── SkillRequest.msg
    │   │   └── EvaluationResult.msg
    │   └── srv/
    │       ├── ExecuteSkill.srv
    │       └── RecordDemo.srv
    │
    ├── mikro_dojo_hal/          # Hardware abstraction
    │   ├── mikro_dojo_hal/
    │   │   ├── vehicle_node.py
    │   │   └── arena_node.py
    │   └── launch/
    │       └── hal.launch.py
    │
    ├── mikro_dojo_perception/   # Perception nodes
    │   ├── mikro_dojo_perception/
    │   │   ├── multi_camera_node.py
    │   │   ├── tracking_node.py
    │   │   └── calibration_node.py
    │   └── launch/
    │       └── perception.launch.py
    │
    ├── mikro_dojo_learning/     # Learning nodes
    │   └── ...
    │
    ├── mikro_dojo_execution/    # Execution nodes
    │   └── ...
    │
    └── mikro_dojo_bringup/      # System launch files
        └── launch/
            ├── arena.launch.py
            ├── vehicle.launch.py
            └── full_system.launch.py
```

---

## Directory Structure (Full Repository)

```
mikro-dojo/
├── README.md
├── ARCHITECTURE.md              # This document
├── CONTRIBUTING.md
├── LICENSE                      # Apache 2.0
├── pyproject.toml              # Python project config
├── setup.py
│
├── docs/
│   ├── design/
│   │   ├── MIKRO_DOJO_Brand_Guidelines.md
│   │   └── RC_CAR_LEARNING_SYSTEM_DESIGN.md
│   ├── setup/
│   │   ├── arena_setup.md
│   │   ├── vehicle_setup.md
│   │   └── jetson_setup.md
│   ├── tutorials/
│   │   ├── first_skill.md
│   │   └── scratch_integration.md
│   └── api/
│       └── reference.md
│
├── mikro_dojo/                  # Core Python package
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── plugins.py
│   │   └── logging.py
│   ├── hal/                     # Hardware abstraction
│   ├── perception/              # Perception system
│   ├── learning/                # Learning engine
│   ├── evaluation/              # Evaluation system
│   ├── execution/               # Runtime execution
│   ├── orchestration/           # Orchestration layer
│   └── gateway/                 # External integrations
│
├── ros2_ws/                     # ROS 2 workspace
│   └── src/
│       ├── mikro_dojo_msgs/
│       ├── mikro_dojo_hal/
│       ├── mikro_dojo_perception/
│       ├── mikro_dojo_learning/
│       ├── mikro_dojo_execution/
│       └── mikro_dojo_bringup/
│
├── config/
│   ├── fleet.yaml               # Vehicle/arena configs
│   ├── skills.yaml              # Skill definitions
│   └── training.yaml            # Training hyperparameters
│
├── scripts/
│   ├── calibrate_arena.py
│   ├── record_demo.py
│   ├── train_skill.py
│   └── deploy_skill.py
│
├── integrations/
│   ├── scratch/
│   │   ├── extension/
│   │   └── server/
│   └── blockly/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.jetson
│   └── docker-compose.yml
│
└── examples/
    ├── basic_driving/
    ├── drift_skill/
    └── multi_car/
```

---

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Robot Framework** | ROS 2 Jazzy | Communication, multi-robot support |
| **Learning** | LeRobot + PyTorch | Imitation learning, policies |
| **RL** | Stable-Baselines3 | Reinforcement learning |
| **VLM** | LLaVA / Qwen-VL + Ollama | Skill evaluation |
| **Perception** | OpenCV + Kalibr | Vision processing |
| **Edge Runtime** | TensorRT + Isaac ROS | Jetson deployment |
| **Data** | Arrow + DVC + MLflow | Data pipeline |
| **API** | FastAPI + WebSocket | External interfaces |
| **Kids Tools** | Scratch 3.0 + Blockly | Visual programming |
| **Visualization** | Foxglove + Grafana | Monitoring |

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up ROS 2 workspace structure
- [ ] Implement HAL interfaces and first vehicle driver
- [ ] Basic multi-camera capture and sync
- [ ] Simple teleoperation demo

### Phase 2: Perception (Weeks 5-8)
- [ ] Multi-camera calibration pipeline
- [ ] Vehicle tracking system
- [ ] Data recording to LeRobot format
- [ ] Arena visualization

### Phase 3: Learning (Weeks 9-14)
- [ ] LeRobot dataset integration
- [ ] Behavioral cloning pipeline
- [ ] VLM evaluation integration
- [ ] First trained skill (forward driving)

### Phase 4: Refinement (Weeks 15-18)
- [ ] RL fine-tuning pipeline
- [ ] Edge deployment on Jetson
- [ ] Safety system implementation
- [ ] Multi-vehicle support

### Phase 5: Integration (Weeks 19-22)
- [ ] REST API and WebSocket server
- [ ] Web UI dashboard
- [ ] Scratch extension (basic blocks)
- [ ] Documentation and examples

---

## Design Principles Summary

1. **Interface-First**: Define abstract interfaces before implementations
2. **Plugin Architecture**: All components are pluggable and replaceable
3. **Configuration-Driven**: Behavior controlled by YAML configs, not code changes
4. **ROS 2 Native**: Use ROS 2 patterns for communication and lifecycle
5. **LeRobot Compatible**: Align with LeRobot data formats and APIs
6. **Progressive Enhancement**: Start simple, add complexity incrementally
7. **Test Everything**: Unit, integration, and simulation tests
8. **Document as You Go**: Keep docs updated with code

---

## Next Steps

1. **Set up repository structure** with core packages
2. **Implement IVehicle interface** and first driver
3. **Create ROS 2 workspace** with message definitions
4. **Build multi-camera capture** node
5. **Integrate LeRobot** dataset format

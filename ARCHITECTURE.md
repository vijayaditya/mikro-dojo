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
    │   └── wltoys_124019.py # Budget option
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

  - id: car_beta
    type: wltoys_124019
    arena: home_arena
    connection:
      type: wifi
      ip: 192.168.1.101
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

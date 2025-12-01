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

For missions requiring high plane capacity and extended operations, the MIKRO-CARRIER is a larger vehicle platform with land, air, and water capabilities:

```python
@VehicleRegistry.register("mikro_carrier")
class MikroCarrier(IVehicle):
    """Large carrier vehicle with integrated plane storage, launcher, and flight capability"""

    SPECIFICATIONS = {
        "scale": "1/5",                    # Larger than standard 1/10
        "length_cm": 80,                   # 80cm long
        "width_cm": 45,                    # 45cm wide
        "height_cm": 35,                   # 35cm tall (with launcher)
        "weight_kg": 8.0,                  # 8kg base weight
        "payload_kg": 5.0,                 # 5kg payload capacity
        "max_ground_speed_kmh": 40,        # 40 km/h on ground
        "max_flight_speed_mph": 100,       # 100 mph in flight mode!
        "drive": "6WD",                    # 6-wheel drive for stability
        "flight_system": "quad_turbine",   # 4x ducted fan turbines
        "battery": "6S 10000mAh LiPo",     # Large battery
        "runtime_ground_minutes": 45,      # 45 min ground runtime
        "runtime_flight_minutes": 15,      # 15 min flight runtime
    }

    FEATURES = [
        "plane_storage_bay",       # Rear cargo bay for 10-20 planes
        "roof_launcher",           # Electromagnetic catapult launcher
        "front_camera",            # Wide-angle stereo camera
        "voice_control",           # Dictation control system
        "gps_navigation",          # Outdoor GPS support
        "4g_connectivity",         # Remote operation capability
        "flight_system",           # Quad turbine flight @ 100mph
        "crash_protection",        # Deployable safety armor layers
        "water_landing",           # Inflatable emergency raft
        "dual_nerf_turrets",       # 2x retractable nerf guns in headlights
        "booster_system",          # Speed boost: 80km/h ground, 150mph air!
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
| Ground Speed | 40 km/h (80 km/h boosted!) |
| **Flight Speed** | **100 mph (150 mph boosted!)** |
| Flight System | Quad ducted fan turbines |
| Battery | 6S 10000mAh LiPo |
| Ground Runtime | 45 minutes |
| Flight Runtime | 15 minutes |
| Plane Capacity | 10-20 planes |
| Compute | Jetson AGX Orin |
| Safety | Crash armor + Water raft |
| **Nerf Turrets** | **2x (60 darts total)** |

**Why Bigger?**
- More space for plane storage magazine (10-20 planes)
- Larger battery for extended missions
- More powerful motors to carry payload
- Room for advanced compute (Jetson AGX Orin)
- Better stability for launching planes while moving
- Space for voice control microphone array
- **Room for flight turbines and safety systems**

---

### Module 1.6: Flight System

**Flying Capabilities**:
The MIKRO-CARRIER can transform from ground vehicle to flying vehicle!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class FlightMode(Enum):
    GROUND = "ground"           # Wheels on ground, turbines off
    HOVER = "hover"             # Stationary in air
    CRUISE = "cruise"           # Forward flight
    HIGH_SPEED = "high_speed"   # Maximum 100mph flight
    LANDING = "landing"         # Descending to land

@dataclass
class FlightState:
    mode: FlightMode
    altitude_meters: float      # Current altitude
    speed_mph: float            # Current airspeed
    battery_percent: float      # Remaining battery
    turbine_rpm: list[int]      # RPM for each of 4 turbines

class IFlightSystem(ABC):
    """Interface for vehicle flight capabilities"""

    @abstractmethod
    def takeoff(self, target_altitude: float) -> bool:
        """Launch into the air"""
        ...

    @abstractmethod
    def land(self) -> bool:
        """Return to ground safely"""
        ...

    @abstractmethod
    def set_altitude(self, meters: float) -> None:
        """Adjust flight altitude"""
        ...

    @abstractmethod
    def set_speed(self, mph: float) -> None:
        """Set flight speed (max 100mph)"""
        ...

    @abstractmethod
    def get_flight_state(self) -> FlightState:
        """Get current flight status"""
        ...

    @abstractmethod
    def emergency_land(self) -> None:
        """Immediate safe landing"""
        ...
```

**Flight System Hardware**:
- **Propulsion**: 4x high-power ducted fan turbines (foldable)
- **Max Speed**: 100 mph (160 km/h)
- **Max Altitude**: 120 meters (400 feet)
- **Turbine Power**: 2000W per turbine (8000W total)
- **Transition Time**: 3 seconds ground-to-air
- **Stabilization**: 6-axis gyro + GPS + barometer
- **Turbine Placement**: 4 corners, fold flat when on ground

**Flight Modes**:
| Mode | Speed | Use Case |
|------|-------|----------|
| Hover | 0 mph | Stationary observation |
| Cruise | 30-50 mph | Normal flight |
| High Speed | 100 mph | Rapid transit / chase |
| Landing | 5-10 mph | Safe descent |

**Voice Commands for Flight**:
- "Take off" - Launch into hover
- "Fly forward" - Start moving
- "Go faster" / "Full speed" - Accelerate to 100mph
- "Land" - Begin landing sequence
- "Emergency land" - Immediate safe landing

---

### Module 1.7: Safety Systems

**Crash Landing Protection**:
If the vehicle detects a crash landing, it deploys protective armor layers!

```python
class CrashSeverity(Enum):
    MINOR = "minor"           # Small bump, no protection needed
    MODERATE = "moderate"     # Deploy soft bumpers
    SEVERE = "severe"         # Deploy full armor shell
    CRITICAL = "critical"     # Deploy all protection + emergency systems

@dataclass
class ProtectionStatus:
    layers_deployed: int       # How many armor layers active (0-3)
    armor_integrity: float     # 0-100% condition
    impact_absorbed: float     # Joules absorbed
    reusable: bool             # Can armor be retracted?

class ICrashProtection(ABC):
    """Interface for crash landing safety system"""

    @abstractmethod
    def detect_impact(self) -> CrashSeverity:
        """Detect incoming crash and severity"""
        ...

    @abstractmethod
    def deploy_protection(self, severity: CrashSeverity) -> bool:
        """Deploy appropriate armor layers"""
        ...

    @abstractmethod
    def get_status(self) -> ProtectionStatus:
        """Get protection system status"""
        ...

    @abstractmethod
    def retract_armor(self) -> bool:
        """Retract armor layers after safe landing"""
        ...

    @abstractmethod
    def reset_system(self) -> None:
        """Reset protection system for reuse"""
        ...
```

**Crash Protection Hardware**:
- **Layer 1 - Foam Bumpers**: Soft foam extends around vehicle (minor impacts)
- **Layer 2 - Airbag Shell**: Inflatable airbags surround the body (moderate impacts)
- **Layer 3 - Rigid Armor**: Hard shell plates deploy from sides (severe impacts)
- **Sensors**: Accelerometers detect imminent crash 100ms before impact
- **Deployment Speed**: Full armor in 50 milliseconds
- **Reusable**: Foam and airbags retract; armor plates reset

**Protection Layers**:
| Layer | Type | Protection | Deployment Trigger |
|-------|------|------------|-------------------|
| 1 | Foam Bumpers | Scratches, small bumps | <2G impact detected |
| 2 | Airbag Shell | Moderate crashes | 2-5G impact detected |
| 3 | Rigid Armor | Hard crashes | >5G impact detected |

---

**Water Landing System**:
If the vehicle lands in water, an inflatable raft deploys from the bottom!

```python
@dataclass
class RaftStatus:
    deployed: bool              # Is raft inflated?
    inflation_percent: float    # 0-100% inflated
    buoyancy_kg: float          # Weight raft can support
    gps_beacon_active: bool     # Rescue beacon on?

class IWaterLanding(ABC):
    """Interface for water landing emergency system"""

    @abstractmethod
    def detect_water(self) -> bool:
        """Detect water contact"""
        ...

    @abstractmethod
    def deploy_raft(self) -> bool:
        """Inflate emergency raft from bottom"""
        ...

    @abstractmethod
    def activate_beacon(self) -> None:
        """Turn on GPS rescue beacon"""
        ...

    @abstractmethod
    def get_raft_status(self) -> RaftStatus:
        """Get raft deployment status"""
        ...

    @abstractmethod
    def deflate_raft(self) -> bool:
        """Deflate and retract raft after rescue"""
        ...
```

**Water Landing Hardware**:
- **Raft Location**: Folded in waterproof compartment under chassis
- **Raft Size**: Inflates to 100cm × 60cm × 15cm
- **Buoyancy**: Supports up to 15kg (vehicle + payload)
- **Inflation**: CO2 cartridge, inflates in 2 seconds
- **Material**: Puncture-resistant PVC
- **Beacon**: GPS + flashing LED for recovery
- **Water Sensors**: 4 contact sensors on underside

**Water Landing Sequence**:
1. Water contact detected by sensors
2. Raft deploys and inflates in 2 seconds
3. Vehicle floats safely on water
4. GPS beacon activates for recovery
5. Bright LED flashes for visibility
6. Motors shut down to prevent water damage

**Voice Commands for Safety**:
- "Deploy raft" - Manual raft deployment
- "Activate beacon" - Turn on rescue signal
- "Armor up" - Deploy all crash protection
- "Safety status" - Report protection system status

### Module 1.8: Booster System

**Nitro Boost for Land and Air!**
Activate the booster to go even faster on the ground or in the air!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class BoostMode(Enum):
    OFF = "off"                 # Normal speed
    GROUND_BOOST = "ground"     # Land speed boost
    AIR_BOOST = "air"           # Flight speed boost
    MAXIMUM = "maximum"         # All boosters at full power

@dataclass
class BoostStatus:
    active: bool                # Is boost currently on?
    mode: BoostMode             # Current boost mode
    fuel_percent: float         # Remaining boost fuel (0-100)
    temperature_celsius: float  # Booster temperature
    cooldown_seconds: float     # Time until boost available again

class IBoosterSystem(ABC):
    """Interface for speed booster system"""

    @abstractmethod
    def activate_boost(self, mode: BoostMode) -> bool:
        """Activate speed boost"""
        ...

    @abstractmethod
    def deactivate_boost(self) -> None:
        """Turn off boost"""
        ...

    @abstractmethod
    def get_status(self) -> BoostStatus:
        """Get current boost status"""
        ...

    @abstractmethod
    def refuel(self) -> None:
        """Refill boost fuel tank"""
        ...

    @abstractmethod
    def get_current_speed_multiplier(self) -> float:
        """Get speed increase factor (1.0 = normal, 2.0 = double)"""
        ...
```

**Booster Hardware**:
- **Type**: Electric turbo boost + afterburner
- **Ground Boost**: Activates extra motor power + rear thruster
- **Air Boost**: Overdrives flight turbines + auxiliary jets
- **Fuel**: Rechargeable boost capacitor bank
- **Boost Duration**: 10 seconds at full power
- **Cooldown**: 30 seconds between boosts
- **Visual Effect**: Blue flames from rear exhausts!

**Speed Boost Specifications**:
| Mode | Normal Speed | Boosted Speed | Increase |
|------|--------------|---------------|----------|
| Ground | 40 km/h | **80 km/h** | 2x faster! |
| Air Cruise | 50 mph | **100 mph** | 2x faster! |
| Air Max | 100 mph | **150 mph** | 1.5x faster! |

**Boost System Hardware**:
- **Capacitor Bank**: High-discharge LiPo cells for instant power
- **Ground Thruster**: Rear-mounted electric ducted fan
- **Air Afterburner**: Auxiliary jet assist on flight turbines
- **Cooling**: Active liquid cooling to prevent overheating
- **Exhaust**: LED-lit blue flame effect pipes
- **Sound**: Speaker plays turbo/jet sounds during boost

**Boost Fuel Gauge**:
| Level | Status | Boost Available |
|-------|--------|-----------------|
| 100-75% | Full | 10 seconds |
| 75-50% | Good | 7 seconds |
| 50-25% | Low | 4 seconds |
| 25-0% | Critical | 2 seconds |
| 0% | Empty | Recharging... |

**Voice Commands for Booster**:
- "Boost" / "Hit it" / "Nitro" - Activate boost
- "Ground boost" - Land speed boost only
- "Air boost" - Flight speed boost only
- "Maximum boost" - All boosters at once
- "Boost off" - Deactivate boost
- "Fuel check" - Report boost fuel level
- "Cool down" - Check cooldown timer

**Booster Configuration**:
```yaml
booster_system:
  enabled: true
  ground_boost:
    enabled: true
    speed_multiplier: 2.0      # Double ground speed
    max_speed_kmh: 80
    thruster_power: 2000       # Watts
  air_boost:
    enabled: true
    speed_multiplier: 1.5      # 50% faster in air
    max_speed_mph: 150
    afterburner_power: 3000    # Watts
  fuel:
    capacity_seconds: 10       # Full boost duration
    recharge_rate: 0.33        # Seconds of fuel per second (30s full recharge)
  safety:
    max_temperature_celsius: 80
    auto_cutoff: true          # Stop boost if overheating
    cooldown_required: true
  effects:
    led_flames: true
    sound_effects: true
    exhaust_color: blue
```

---

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
        ├── voice_control.py       # Dictation/voice control system
        ├── flight_system.py       # Quad turbine flight system (100mph)
        ├── crash_protection.py    # 3-layer crash armor system
        ├── water_landing.py       # Inflatable raft emergency system
        ├── nerf_turret.py         # Retractable nerf gun turret system
        └── booster_system.py      # Speed boost for land and air
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

---

**Dual Nerf Turret System**:
TWO retractable nerf guns that pop out from each side of the headlight area!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class TurretState(Enum):
    RETRACTED = "retracted"     # Hidden inside headlight housing
    DEPLOYING = "deploying"     # Extending out
    READY = "ready"             # Deployed and ready to fire
    FIRING = "firing"           # Currently shooting
    RELOADING = "reloading"     # Loading next dart
    RETRACTING = "retracting"   # Going back inside

@dataclass
class TurretPosition:
    pan_degrees: float          # Left/right aim (-45 to +45)
    tilt_degrees: float         # Up/down aim (-15 to +30)
    deployed: bool              # Is turret out?

@dataclass
class AmmoStatus:
    darts_loaded: int           # Darts in current magazine
    magazines_remaining: int    # Spare magazines
    dart_type: str              # "standard", "elite", "mega"

class INerfTurret(ABC):
    """Interface for retractable nerf gun turret"""

    @abstractmethod
    def deploy(self) -> bool:
        """Extend turret from headlight housing"""
        ...

    @abstractmethod
    def retract(self) -> bool:
        """Hide turret back inside headlight housing"""
        ...

    @abstractmethod
    def aim(self, pan: float, tilt: float) -> None:
        """Aim turret (pan: left/right, tilt: up/down)"""
        ...

    @abstractmethod
    def fire(self, burst_count: int = 1) -> bool:
        """Fire nerf darts (1 = single, 3 = burst, -1 = full auto)"""
        ...

    @abstractmethod
    def reload(self) -> bool:
        """Load next magazine"""
        ...

    @abstractmethod
    def get_ammo_status(self) -> AmmoStatus:
        """Get current ammo count"""
        ...

    @abstractmethod
    def get_position(self) -> TurretPosition:
        """Get turret aim position"""
        ...

    @abstractmethod
    def auto_aim(self, target_coordinates: tuple) -> None:
        """Automatically aim at target using camera"""
        ...
```

**Dual Nerf Turret Hardware**:
- **Quantity**: 2 turrets (LEFT and RIGHT)
- **Location**: Hidden in left and right headlight housings
- **Deployment**: Motorized slide mechanism, deploys in 1.5 seconds
- **Magazine Capacity**: 10 darts per magazine per turret
- **Spare Magazines**: 2 additional magazines per turret (60 darts total!)
- **Fire Rate**: 3 darts per second per turret (6 darts/sec combined!)
- **Range**: Up to 15 meters (50 feet)
- **Aim System**: Each turret: Pan ±45°, Tilt -15° to +30°
- **Targeting**: Camera-assisted auto-aim, can track 2 targets simultaneously
- **Sync Mode**: Both turrets can fire together or independently

**Dual Turret Specifications**:
| Specification | Per Turret | Combined (Both) |
|---------------|------------|-----------------|
| Magazine Size | 10 darts | 20 darts |
| Total Capacity | 30 darts | **60 darts** |
| Fire Rate | 3 darts/sec | **6 darts/sec** |
| Range | 15 meters | 15 meters |
| Pan Range | ±45 degrees | ±45 degrees |
| Tilt Range | -15° to +30° | -15° to +30° |
| Deploy Time | 1.5 seconds | 1.5 seconds |
| Motor | High-torque servo | 2x servos |
| Flywheel | Dual flywheel | 4 flywheels total |

**Firing Modes**:
| Mode | Description | Voice Command |
|------|-------------|---------------|
| Single | One dart from each turret | "Fire" / "Shoot" |
| Burst | 3 rapid darts from each | "Burst fire" |
| Full Auto | Both turrets continuous | "Full auto" / "Unleash" |
| Alternating | Left-right-left-right | "Alternating fire" |
| Left Only | Fire left turret only | "Fire left" |
| Right Only | Fire right turret only | "Fire right" |

**Voice Commands for Dual Nerf Turrets**:
- "Deploy turrets" - Pop out both nerf guns
- "Deploy left" / "Deploy right" - Deploy one turret
- "Retract turrets" - Hide both nerf guns
- "Aim left" / "Aim right" - Pan both turrets
- "Aim up" / "Aim down" - Tilt both turrets
- "Fire" / "Shoot" - Single shot from both
- "Fire left" / "Fire right" - Fire one turret
- "Burst fire" - 3-dart burst from both
- "Full auto" - Empty both magazines
- "Alternating fire" - Left-right pattern
- "Reload" - Reload both magazines
- "Ammo check" - Report remaining darts
- "Auto aim" - Track targets with camera
- "Split targets" - Each turret tracks different target

**Dual Turret Configuration**:
```yaml
nerf_turrets:
  enabled: true
  count: 2
  turret_left:
    location: left_headlight
    magazine_capacity: 10
    total_magazines: 3
  turret_right:
    location: right_headlight
    magazine_capacity: 10
    total_magazines: 3
  dart_type: elite
  sync_mode: together  # "together", "alternating", "independent"
  auto_aim:
    enabled: true
    tracking_camera: front_camera
    multi_target: true  # Track 2 targets at once
    target_detection: color_based
  safety:
    max_range_limit: true
    friendly_fire_prevention: true
    deploy_speed_limit: 5  # Don't fire while moving fast
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

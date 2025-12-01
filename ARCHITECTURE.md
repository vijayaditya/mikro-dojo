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
        "gum_dispenser",           # 50 gumballs, 6 flavors, launch or drop!
        "clone_deployer",          # Deploy mini-clone drones!
        "robot_bodyguards",        # 2 mini humanoid robot bodyguards with nerf guns!
        "candy_dispenser",         # 100 pieces, multiple candy types!
        "speaker_system",          # Play music, sounds, announcements!
        "earpod_case",             # Store and charge wireless earbuds!
        "toilet_feature",          # Mini toilet for emergencies!
        "atm_machine",             # Withdraw cash on the go!
        "vending_machine",         # Snacks and drinks on the go!
        "controller_ai",           # AI assistant on the controller!
        "icee_machine",            # Frozen ICEE slushie dispenser!
        "tv_screen",               # TV screen above the ICEE machine!
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
| Gum Dispenser | 50 gumballs, 6 flavors |
| **Clone Drones** | **4 mini-clones!** |
| **Bodyguard Nerf Guns** | **4 guns, 24 darts!** |
| Candy Dispenser | 100 pieces, 6 types, climate controlled! |
| Speaker System | 40W, Bluetooth, TTS announcements! |
| Earpod Case | Store & charge wireless earbuds! |
| Toilet Feature | Emergency mini toilet with privacy! |
| ATM Machine | Withdraw cash anywhere! |
| Vending Machine | Snacks & drinks dispenser! |
| Controller AI | Smart AI assistant on controller! |
| ICEE Machine | Frozen slushie drinks! |
| TV Screen | Watch shows above the ICEE machine! |

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
        ├── booster_system.py      # Speed boost for land and air
        ├── gum_dispenser.py       # Gum dispenser system
        ├── clone_deployer.py      # Mini-clone drone deployment system
        ├── robot_bodyguards.py    # Mini humanoid robot bodyguards
        ├── candy_dispenser.py     # Candy dispenser with climate control
        ├── speaker_system.py      # Speaker system for music & effects
        ├── earpod_case.py         # Wireless earbud storage & charging
        ├── toilet_feature.py      # Emergency toilet system
        ├── atm_machine.py         # Mobile ATM for cash withdrawals
        ├── vending_machine.py     # Snacks & drinks vending machine
        ├── controller_ai.py       # AI assistant on the controller
        ├── icee_machine.py        # Frozen ICEE slushie dispenser
        └── tv_screen.py           # TV screen above ICEE machine
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

---

**Gum Dispenser System**:
A built-in gum dispenser that can drop or launch gumballs!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class GumFlavor(Enum):
    BUBBLEGUM = "bubblegum"       # Classic pink
    MINT = "mint"                  # Fresh mint
    STRAWBERRY = "strawberry"      # Fruity
    GRAPE = "grape"                # Purple grape
    WATERMELON = "watermelon"      # Summer favorite
    MYSTERY = "mystery"            # Random flavor!

class DispenseMode(Enum):
    DROP = "drop"                  # Gentle drop below vehicle
    LAUNCH = "launch"              # Launch forward
    SCATTER = "scatter"            # Multiple gumballs spread out

@dataclass
class GumInventory:
    total_gumballs: int            # Total gum remaining
    flavors: dict[GumFlavor, int]  # Count per flavor
    dispenser_ready: bool          # Ready to dispense?

class IGumDispenser(ABC):
    """Interface for gum dispenser system"""

    @abstractmethod
    def dispense(self, count: int = 1, mode: DispenseMode = DispenseMode.DROP) -> bool:
        """Dispense gumballs"""
        ...

    @abstractmethod
    def select_flavor(self, flavor: GumFlavor) -> bool:
        """Select which flavor to dispense"""
        ...

    @abstractmethod
    def get_inventory(self) -> GumInventory:
        """Get current gum inventory"""
        ...

    @abstractmethod
    def refill(self, flavor: GumFlavor, count: int) -> None:
        """Refill gumballs"""
        ...

    @abstractmethod
    def set_dispense_mode(self, mode: DispenseMode) -> None:
        """Set how gum is dispensed"""
        ...
```

**Gum Dispenser Hardware**:
- **Location**: Side panel compartment with drop chute
- **Capacity**: 50 gumballs total
- **Gumball Size**: Standard 1-inch (2.5cm) diameter
- **Dispense Rate**: Up to 5 gumballs per second
- **Launch Distance**: Up to 3 meters when launched
- **Flavor Sorting**: Automatic color-based sorting
- **Refill**: Quick-load top hatch

**Gum Dispenser Specifications**:
| Specification | Value |
|---------------|-------|
| Capacity | 50 gumballs |
| Gumball Size | 1 inch (2.5cm) |
| Flavors | 6 different flavors |
| Dispense Rate | 5 per second |
| Launch Range | 3 meters |
| Refill Time | 10 seconds |

**Dispense Modes**:
| Mode | Description | Use Case |
|------|-------------|----------|
| Drop | Gentle release below | Leaving a trail |
| Launch | Forward projection | Sharing with friends |
| Scatter | Multiple directions | Party mode! |

**Voice Commands for Gum Dispenser**:
- "Dispense gum" / "Drop gum" - Release one gumball
- "Gum please" - Polite dispense
- "Bubblegum" / "Mint" / "Grape" - Select flavor
- "Mystery flavor" - Random surprise!
- "Launch gum" - Shoot gumball forward
- "Scatter gum" - Party scatter mode
- "Gum count" - Report remaining gumballs
- "Gum refill needed" - Check if low

**Gum Dispenser Configuration**:
```yaml
gum_dispenser:
  enabled: true
  location: side_panel
  capacity: 50
  gumball_size_cm: 2.5
  flavors:
    bubblegum: 10
    mint: 10
    strawberry: 10
    grape: 10
    watermelon: 5
    mystery: 5
  dispense_settings:
    default_mode: drop
    launch_power: 50          # Percent power for launch
    scatter_count: 5          # Gumballs per scatter
  alerts:
    low_warning: 10           # Warn when 10 left
    empty_warning: true
```

---

**Clone Deployer System**:
Deploy mini-clone drones that look and act like tiny versions of the MIKRO-CARRIER!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class CloneSize(Enum):
    MICRO = "micro"            # 10cm - tiny scout
    MINI = "mini"              # 20cm - standard clone
    SMALL = "small"            # 30cm - capable clone

class CloneMode(Enum):
    FOLLOW = "follow"          # Follow the main vehicle
    SCOUT = "scout"            # Explore ahead
    GUARD = "guard"            # Circle and protect
    SWARM = "swarm"            # Coordinated group behavior
    MIRROR = "mirror"          # Copy main vehicle's movements

@dataclass
class CloneStatus:
    clone_id: int              # Unique clone identifier
    battery_percent: float     # Clone's battery level
    distance_meters: float     # Distance from main vehicle
    mode: CloneMode            # Current behavior mode
    active: bool               # Is clone deployed?

@dataclass
class CloneFleet:
    total_clones: int          # How many clones available
    deployed_clones: int       # How many currently out
    clones: list[CloneStatus]  # Status of each clone

class ICloneDeployer(ABC):
    """Interface for mini-clone deployment system"""

    @abstractmethod
    def deploy_clone(self, mode: CloneMode = CloneMode.FOLLOW) -> int:
        """Deploy a clone, returns clone ID"""
        ...

    @abstractmethod
    def deploy_all(self, mode: CloneMode = CloneMode.SWARM) -> list[int]:
        """Deploy all available clones"""
        ...

    @abstractmethod
    def recall_clone(self, clone_id: int) -> bool:
        """Recall specific clone back to carrier"""
        ...

    @abstractmethod
    def recall_all(self) -> bool:
        """Recall all clones"""
        ...

    @abstractmethod
    def set_clone_mode(self, clone_id: int, mode: CloneMode) -> None:
        """Change a clone's behavior mode"""
        ...

    @abstractmethod
    def get_fleet_status(self) -> CloneFleet:
        """Get status of all clones"""
        ...

    @abstractmethod
    def swarm_command(self, command: str) -> None:
        """Send command to all deployed clones"""
        ...
```

**Clone Drone Hardware**:
- **Capacity**: 4 mini-clone drones stored in carrier
- **Clone Size**: 20cm × 12cm × 10cm each
- **Clone Weight**: 150g each
- **Clone Speed**: Ground 20 km/h, Air 40 mph
- **Clone Flight**: Yes! Mini quad-rotors
- **Clone Battery**: 10 minutes flight time
- **Clone Camera**: 720p streaming back to carrier
- **Docking**: Magnetic auto-dock for recharging

**Clone Specifications**:
| Specification | Per Clone | All 4 Clones |
|---------------|-----------|--------------|
| Size | 20cm long | - |
| Weight | 150g | 600g total |
| Ground Speed | 20 km/h | - |
| Air Speed | 40 mph | - |
| Flight Time | 10 minutes | - |
| Camera | 720p | 4 viewpoints! |
| Range | 100 meters | - |

**Clone Behavior Modes**:
| Mode | Description | Use Case |
|------|-------------|----------|
| Follow | Stay behind main vehicle | Escort formation |
| Scout | Fly ahead and report back | Exploration |
| Guard | Circle around carrier | Protection |
| Swarm | Coordinated group patterns | Show off! |
| Mirror | Copy carrier's movements | Synchronized action |

**Voice Commands for Clone Deployer**:
- "Deploy clone" - Launch one clone
- "Deploy all clones" - Launch entire fleet
- "Recall clone" / "Come back" - Recall clones
- "Clones follow me" - Follow mode
- "Clones scout ahead" - Scout mode
- "Clones guard" - Guard formation
- "Clones swarm" - Swarm pattern
- "Clones mirror" - Mirror movements
- "Clone status" - Report fleet status
- "Clone 1 scout" / "Clone 2 follow" - Individual commands

**Clone Deployer Configuration**:
```yaml
clone_deployer:
  enabled: true
  capacity: 4
  clone_specs:
    size_cm: 20
    weight_g: 150
    ground_speed_kmh: 20
    air_speed_mph: 40
    flight_time_minutes: 10
    camera_resolution: 720p
    range_meters: 100
  docking:
    location: rear_bay
    auto_dock: true
    charge_time_minutes: 15
  behavior:
    default_mode: follow
    formation_spacing_meters: 2
    swarm_pattern: diamond
    collision_avoidance: true
  communication:
    frequency: "2.4GHz"
    video_streaming: true
    telemetry_rate_hz: 10
```

**Clone Fleet Formations**:
```
DIAMOND:          LINE:            SPREAD:
    1               1 2 3 4          1       4
   2 3                               2   C   3
    4
    C                   C            (C = Carrier)
```

---

**Robot Bodyguard System**:
Two mini humanoid robot bodyguards that deploy from the sides of the vehicle - each armed with nerf guns in both hands!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class BodyguardPosition(Enum):
    LEFT = "left"              # Left side of vehicle
    RIGHT = "right"            # Right side of vehicle

class BodyguardStance(Enum):
    DOCKED = "docked"          # Stored inside vehicle
    STANDING = "standing"      # Standing guard
    WALKING = "walking"        # Moving alongside
    DEFENDING = "defending"    # Active defense pose
    DANCING = "dancing"        # Fun mode!
    WAVING = "waving"          # Friendly greeting
    AIMING = "aiming"          # Nerf guns raised and ready!
    FIRING = "firing"          # Shooting nerf darts!

@dataclass
class NerfGunStatus:
    darts_loaded: int          # Darts in this gun (max 6)
    ready_to_fire: bool        # Gun ready?

@dataclass
class BodyguardStatus:
    position: BodyguardPosition  # Left or right
    stance: BodyguardStance      # Current pose/action
    battery_percent: float       # Battery level
    distance_from_vehicle: float # How far from carrier
    arm_position: str            # "down", "up", "crossed", "waving", "aiming"
    left_hand_nerf: NerfGunStatus   # Left hand nerf gun
    right_hand_nerf: NerfGunStatus  # Right hand nerf gun
    total_darts: int             # Total darts remaining

class IRobotBodyguard(ABC):
    """Interface for mini humanoid robot bodyguards with nerf guns"""

    @abstractmethod
    def deploy(self, position: BodyguardPosition) -> bool:
        """Deploy bodyguard from side compartment"""
        ...

    @abstractmethod
    def deploy_both(self) -> bool:
        """Deploy both bodyguards"""
        ...

    @abstractmethod
    def dock(self, position: BodyguardPosition) -> bool:
        """Return bodyguard to vehicle"""
        ...

    @abstractmethod
    def dock_both(self) -> bool:
        """Dock both bodyguards"""
        ...

    @abstractmethod
    def set_stance(self, position: BodyguardPosition, stance: BodyguardStance) -> None:
        """Set bodyguard pose/action"""
        ...

    @abstractmethod
    def walk_with_vehicle(self) -> None:
        """Bodyguards walk alongside moving vehicle"""
        ...

    @abstractmethod
    def get_status(self, position: BodyguardPosition) -> BodyguardStatus:
        """Get bodyguard status"""
        ...

    @abstractmethod
    def perform_action(self, action: str) -> None:
        """Special actions: wave, salute, dance, high-five"""
        ...

    # NERF GUN METHODS
    @abstractmethod
    def aim_guns(self, position: BodyguardPosition, target: tuple) -> None:
        """Aim nerf guns at target coordinates"""
        ...

    @abstractmethod
    def fire_nerf(self, position: BodyguardPosition, hand: str = "both") -> bool:
        """Fire nerf gun(s) - hand: 'left', 'right', or 'both'"""
        ...

    @abstractmethod
    def fire_all_guards(self) -> bool:
        """All bodyguards fire all nerf guns at once!"""
        ...

    @abstractmethod
    def reload_nerf(self, position: BodyguardPosition) -> None:
        """Reload nerf guns (must dock to reload)"""
        ...

    @abstractmethod
    def get_ammo_status(self) -> dict:
        """Get ammo count for all bodyguard nerf guns"""
        ...
```

**Robot Bodyguard Hardware**:
- **Quantity**: 2 bodyguards (LEFT and RIGHT)
- **Storage**: Side compartments that slide open
- **Height**: 25cm tall (humanoid form)
- **Weight**: 350g each (including nerf guns)
- **Joints**: 12 servo motors per robot (articulated limbs)
- **Walking Speed**: Matches vehicle up to 5 km/h
- **Battery**: 30 minutes active time
- **Recharge**: Auto-charge when docked
- **NERF GUNS**: 2 mini nerf guns per bodyguard (one in each hand!)
- **Darts Per Gun**: 6 darts each
- **Total Darts**: 24 darts (4 guns × 6 darts)
- **Fire Rate**: 2 darts/second per gun

**Bodyguard Specifications**:
| Specification | Per Bodyguard | Both |
|---------------|---------------|------|
| Height | 25cm | - |
| Weight | 350g | 700g |
| Joints | 12 servos | 24 servos |
| Walking Speed | 5 km/h | - |
| Battery Life | 30 minutes | - |
| Arm Reach | 8cm | - |
| Head Movement | Pan/tilt | - |
| **Nerf Guns** | **2 (both hands)** | **4 guns total!** |
| **Darts Per Gun** | **6 darts** | **24 darts total!** |
| **Fire Rate** | **2 darts/sec/gun** | **8 darts/sec max!** |
| **Range** | **5 meters** | - |

**Bodyguard Actions**:
| Action | Description | Voice Command |
|--------|-------------|---------------|
| Stand Guard | Stand at attention beside vehicle | "Guards stand" |
| Walk | Walk alongside moving vehicle | "Guards walk" |
| Wave | Friendly wave gesture | "Guards wave" |
| Salute | Military salute | "Guards salute" |
| Dance | Fun dance moves | "Guards dance" |
| High-Five | Reach out for high-five | "High five" |
| Defend | Defensive stance | "Guards defend" |
| Cross Arms | Cool crossed-arms pose | "Guards cross arms" |
| **Aim** | **Raise nerf guns and aim** | **"Guards aim"** |
| **Fire** | **Shoot nerf darts!** | **"Guards fire!"** |
| **Dual Wield** | **Fire both guns rapidly** | **"Guards unleash!"** |

**Voice Commands for Robot Bodyguards**:
- "Deploy guards" - Both bodyguards emerge from sides
- "Deploy left guard" / "Deploy right guard" - One at a time
- "Guards return" / "Dock guards" - Return to vehicle
- "Guards stand" - Standing guard pose
- "Guards walk with me" - Walk alongside
- "Guards wave" - Friendly wave
- "Guards dance" - Party mode!
- "Guards salute" - Military salute
- "Guards high-five" - High-five pose
- "Guard status" - Check battery and status
- **"Guards aim"** - Raise nerf guns ready to fire
- **"Guards fire!"** - Shoot nerf darts
- **"Guards unleash!"** - Rapid fire all guns!
- **"Left guard fire" / "Right guard fire"** - One guard fires
- **"Ammo check"** - Check dart count
- **"Reload guards"** - Dock and reload nerf guns

**Bodyguard Configuration**:
```yaml
robot_bodyguards:
  enabled: true
  count: 2
  left_guard:
    storage: left_side_panel
    color: blue
    personality: serious
  right_guard:
    storage: right_side_panel
    color: red
    personality: friendly
  specs:
    height_cm: 25
    weight_g: 350              # Heavier with nerf guns
    joints: 12
    walk_speed_kmh: 5
    battery_minutes: 30
  nerf_guns:                   # NEW: Nerf gun configuration!
    guns_per_guard: 2          # One in each hand
    darts_per_gun: 6
    total_darts: 24            # 4 guns × 6 darts
    fire_rate_per_second: 2
    range_meters: 5
    auto_aim: true             # Camera-assisted targeting
    reload_on_dock: true       # Auto-reload when docked
  behaviors:
    auto_deploy_on_stop: false
    walk_with_vehicle: true
    max_distance_meters: 3
    return_on_low_battery: true
  actions:
    wave_duration_seconds: 3
    dance_moves: ["robot", "shuffle", "spin"]
    salute_style: military
    fire_pose: dual_wield      # Both arms extended
```

**Bodyguard Formation (Armed!)**:
```
         [CARRIER]
   🤖🔫          🔫🤖
  LEFT            RIGHT
 GUARD            GUARD
  🔫                🔫

  4 Nerf Guns Total!
  24 Darts Ready!
```

---

**Candy Dispenser System**:
A sweet treat dispenser that can drop or launch various candies!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class CandyType(Enum):
    CHOCOLATE = "chocolate"        # Mini chocolate pieces
    GUMMY_BEARS = "gummy_bears"    # Gummy bear candies
    LOLLIPOP = "lollipop"          # Mini lollipops
    HARD_CANDY = "hard_candy"      # Wrapped hard candies
    SOUR_CANDY = "sour_candy"      # Sour gummy worms
    JELLY_BEANS = "jelly_beans"    # Assorted jelly beans

class CandyDispenseMode(Enum):
    DROP = "drop"                  # Gentle drop below vehicle
    TOSS = "toss"                  # Gentle toss forward
    LAUNCH = "launch"              # Launch with more power
    RAIN = "rain"                  # Multiple candies scattered

@dataclass
class CandyInventory:
    total_pieces: int              # Total candy remaining
    candy_types: dict[CandyType, int]  # Count per type
    dispenser_ready: bool          # Ready to dispense?
    temperature_ok: bool           # Chocolate not melted?

class ICandyDispenser(ABC):
    """Interface for candy dispenser system"""

    @abstractmethod
    def dispense(self, count: int = 1, mode: CandyDispenseMode = CandyDispenseMode.DROP) -> bool:
        """Dispense candies"""
        ...

    @abstractmethod
    def select_candy(self, candy_type: CandyType) -> bool:
        """Select which candy type to dispense"""
        ...

    @abstractmethod
    def get_inventory(self) -> CandyInventory:
        """Get current candy inventory"""
        ...

    @abstractmethod
    def refill(self, candy_type: CandyType, count: int) -> None:
        """Refill candies"""
        ...

    @abstractmethod
    def set_dispense_mode(self, mode: CandyDispenseMode) -> None:
        """Set how candy is dispensed"""
        ...

    @abstractmethod
    def check_temperature(self) -> bool:
        """Check if compartment is cool enough for chocolate"""
        ...
```

**Candy Dispenser Hardware**:
- **Location**: Rear panel compartment with climate control
- **Capacity**: 100 candy pieces total
- **Compartments**: 6 separate compartments (one per candy type)
- **Dispense Rate**: Up to 10 pieces per second
- **Launch Distance**: Up to 5 meters when launched
- **Climate Control**: Cooling system to keep chocolate from melting
- **Refill**: Quick-load drawer system

**Candy Dispenser Specifications**:
| Specification | Value |
|---------------|-------|
| Capacity | 100 pieces |
| Candy Types | 6 different types |
| Compartments | 6 separate |
| Dispense Rate | 10 per second |
| Launch Range | 5 meters |
| Climate Control | Active cooling |
| Temperature Range | 15-20°C (chocolate safe) |

**Dispense Modes**:
| Mode | Description | Use Case |
|------|-------------|----------|
| Drop | Gentle release below | Leaving a trail |
| Toss | Soft forward throw | Sharing nearby |
| Launch | Powered projection | Sharing far away |
| Rain | Multiple candies | Party celebration! |

**Voice Commands for Candy Dispenser**:
- "Dispense candy" / "Drop candy" - Release one candy
- "Candy please" - Polite dispense
- "Chocolate" / "Gummy bears" / "Lollipop" - Select type
- "Jelly beans" / "Sour candy" / "Hard candy" - Select type
- "Launch candy" - Shoot candy forward
- "Candy rain" - Party mode!
- "Candy count" - Report remaining candies
- "Chocolate check" - Check if chocolate is okay

**Candy Dispenser Configuration**:
```yaml
candy_dispenser:
  enabled: true
  location: rear_panel
  capacity: 100
  climate_control:
    enabled: true
    target_temp_celsius: 18
    max_temp_celsius: 22
  compartments:
    chocolate: 20
    gummy_bears: 20
    lollipop: 15
    hard_candy: 15
    sour_candy: 15
    jelly_beans: 15
  dispense_settings:
    default_mode: drop
    launch_power: 60
    rain_count: 10
  alerts:
    low_warning: 15
    temperature_warning: true
    empty_warning: true
```

---

**Speaker System**:
A powerful speaker system for music, sound effects, and announcements!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class SoundType(Enum):
    MUSIC = "music"                # Play music tracks
    EFFECT = "effect"              # Sound effects
    VOICE = "voice"                # Voice announcements
    HORN = "horn"                  # Vehicle horn sounds
    ALARM = "alarm"                # Warning/alert sounds

class SpeakerMode(Enum):
    NORMAL = "normal"              # Standard volume
    LOUD = "loud"                  # Maximum volume
    QUIET = "quiet"                # Low volume
    BASS_BOOST = "bass_boost"      # Extra bass
    SURROUND = "surround"          # Surround sound effect

@dataclass
class SpeakerStatus:
    playing: bool                  # Currently playing audio?
    volume_percent: int            # Current volume (0-100)
    track_name: str                # Currently playing track
    battery_draw_watts: float      # Power consumption
    mode: SpeakerMode              # Current speaker mode

class ISpeakerSystem(ABC):
    """Interface for speaker system"""

    @abstractmethod
    def play_music(self, track: str) -> bool:
        """Play a music track"""
        ...

    @abstractmethod
    def play_sound_effect(self, effect: str) -> bool:
        """Play a sound effect"""
        ...

    @abstractmethod
    def announce(self, message: str) -> bool:
        """Text-to-speech announcement"""
        ...

    @abstractmethod
    def honk(self, pattern: str = "short") -> None:
        """Sound the horn"""
        ...

    @abstractmethod
    def set_volume(self, percent: int) -> None:
        """Set speaker volume (0-100)"""
        ...

    @abstractmethod
    def set_mode(self, mode: SpeakerMode) -> None:
        """Set speaker mode"""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop all audio"""
        ...

    @abstractmethod
    def get_status(self) -> SpeakerStatus:
        """Get speaker status"""
        ...
```

**Speaker System Hardware**:
- **Speakers**: 2x 10W full-range speakers + 1x 20W subwoofer
- **Total Power**: 40W RMS
- **Frequency Range**: 60Hz - 20kHz
- **Amplifier**: Class D digital amplifier
- **Storage**: 8GB for music and sound effects
- **Bluetooth**: Yes, for streaming from phone
- **Location**: Side-mounted speakers, rear subwoofer

**Speaker Specifications**:
| Specification | Value |
|---------------|-------|
| Total Power | 40W RMS |
| Speakers | 2x 10W + 1x 20W sub |
| Frequency Range | 60Hz - 20kHz |
| Max Volume | 95 dB |
| Bluetooth | Yes (5.0) |
| Storage | 8GB onboard |
| Waterproof | IPX4 splash resistant |

**Pre-loaded Sound Effects**:
| Category | Examples |
|----------|----------|
| Horns | Car horn, truck horn, musical horn |
| Engines | Sports car, race car, jet engine |
| Fun | Laser, explosion, victory fanfare |
| Animals | Dog bark, lion roar, eagle |
| Music | Victory theme, action music, chase music |
| Alerts | Warning beep, alarm, siren |

**Voice Commands for Speaker**:
- "Play music" - Start playing music
- "Play [track name]" - Play specific track
- "Stop music" / "Quiet" - Stop playing
- "Volume up" / "Louder" - Increase volume
- "Volume down" / "Softer" - Decrease volume
- "Maximum volume" - Full volume
- "Honk" / "Horn" - Sound the horn
- "Play engine sound" - Engine sound effect
- "Victory music" - Play celebration music
- "Announce [message]" - Text-to-speech
- "Bass boost" - Enable bass boost mode

**Speaker Configuration**:
```yaml
speaker_system:
  enabled: true
  speakers:
    left: 10W
    right: 10W
    subwoofer: 20W
  amplifier: class_d
  default_volume: 50
  max_volume: 100
  bluetooth:
    enabled: true
    name: "MIKRO-CARRIER-SPEAKER"
  storage_gb: 8
  preloaded_sounds:
    horns: ["car", "truck", "musical", "train"]
    engines: ["sports_car", "race_car", "jet", "spaceship"]
    effects: ["laser", "explosion", "victory", "coin"]
    music: ["action", "chase", "victory", "chill"]
  text_to_speech:
    enabled: true
    voice: "default"
    language: "en-US"
  horn_patterns:
    short: [200]              # 200ms honk
    long: [1000]              # 1 second honk
    double: [200, 100, 200]   # Two short honks
    shave_and_haircut: [200, 100, 200, 100, 400, 200, 400]
```

---

**Earpod Case System**:
A built-in case to store and charge your wireless earbuds while on the go!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class EarpodState(Enum):
    EMPTY = "empty"                # No earbuds in case
    STORED = "stored"              # Earbuds stored and charging
    READY = "ready"                # Fully charged and ready
    DISPENSING = "dispensing"      # Ejecting earbuds

@dataclass
class EarpodStatus:
    earbuds_present: bool          # Are earbuds in the case?
    case_battery_percent: int      # Case battery level (0-100)
    earpod_battery_percent: int    # Earpod battery level (0-100)
    charging: bool                 # Currently charging earbuds?
    case_open: bool                # Is the case lid open?

class IEarpodCase(ABC):
    """Interface for earpod case system"""

    @abstractmethod
    def open_case(self) -> bool:
        """Open the earpod case lid"""
        ...

    @abstractmethod
    def close_case(self) -> bool:
        """Close the earpod case lid"""
        ...

    @abstractmethod
    def eject_earbuds(self) -> bool:
        """Pop earbuds up for easy grabbing"""
        ...

    @abstractmethod
    def get_status(self) -> EarpodStatus:
        """Get case and earpod status"""
        ...

    @abstractmethod
    def start_charging(self) -> None:
        """Begin charging earbuds"""
        ...

    @abstractmethod
    def pair_bluetooth(self) -> bool:
        """Pair earbuds with vehicle's Bluetooth"""
        ...
```

**Earpod Case Hardware**:
- **Location**: Dashboard compartment with pop-up lid
- **Compatibility**: Universal - fits most wireless earbuds
- **Case Battery**: 500mAh (charges earbuds 3x)
- **Charging Port**: USB-C input for case charging
- **Wireless Charging**: Qi-compatible charging pad
- **Eject Mechanism**: Spring-loaded pop-up
- **LED Indicators**: Battery level lights

**Earpod Case Specifications**:
| Specification | Value |
|---------------|-------|
| Case Battery | 500mAh |
| Earbud Charges | 3x full charges |
| Charging Type | Wireless Qi + USB-C |
| Compatibility | Universal |
| Eject Mechanism | Spring pop-up |
| LED Indicators | 4 battery lights |

**Voice Commands for Earpod Case**:
- "Open earpod case" - Pop open the lid
- "Close earpod case" - Close the lid
- "Eject earbuds" - Pop earbuds up for grabbing
- "Earpod battery" - Check earbud battery level
- "Case battery" - Check case battery level
- "Pair earbuds" - Connect to vehicle Bluetooth

**Earpod Case Configuration**:
```yaml
earpod_case:
  enabled: true
  location: dashboard
  case_battery_mah: 500
  charging:
    wireless_qi: true
    usb_c: true
  compatibility: universal
  auto_pair: true
  led_indicators: 4
  eject_style: spring_popup
```

---

**Toilet Feature**:
An emergency mini toilet that deploys from the vehicle for those urgent situations!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class ToiletState(Enum):
    STOWED = "stowed"              # Hidden inside vehicle
    DEPLOYING = "deploying"        # Extending out
    READY = "ready"                # Ready for use
    IN_USE = "in_use"              # Currently occupied
    CLEANING = "cleaning"          # Self-cleaning cycle
    RETRACTING = "retracting"      # Going back inside

class PrivacyMode(Enum):
    NONE = "none"                  # No privacy screen
    PARTIAL = "partial"            # Side screens only
    FULL = "full"                  # Full enclosure deployed

@dataclass
class ToiletStatus:
    state: ToiletState             # Current toilet state
    privacy_mode: PrivacyMode      # Privacy screen status
    waste_tank_percent: int        # Waste tank fill level
    water_tank_percent: int        # Fresh water level
    sanitizer_level: int           # Sanitizer remaining
    last_cleaned: str              # Timestamp of last clean

class IToiletFeature(ABC):
    """Interface for emergency toilet system"""

    @abstractmethod
    def deploy(self) -> bool:
        """Deploy toilet from vehicle"""
        ...

    @abstractmethod
    def retract(self) -> bool:
        """Retract toilet back into vehicle"""
        ...

    @abstractmethod
    def set_privacy(self, mode: PrivacyMode) -> None:
        """Deploy privacy screens"""
        ...

    @abstractmethod
    def flush(self) -> bool:
        """Flush and clean toilet"""
        ...

    @abstractmethod
    def auto_clean(self) -> None:
        """Run automatic cleaning cycle"""
        ...

    @abstractmethod
    def get_status(self) -> ToiletStatus:
        """Get toilet system status"""
        ...

    @abstractmethod
    def refill_supplies(self) -> None:
        """Refill water and sanitizer"""
        ...
```

**Toilet Hardware**:
- **Location**: Rear compartment with slide-out deployment
- **Seat Type**: Compact folding seat
- **Privacy**: Pop-up privacy screens (3-sided enclosure)
- **Waste Tank**: 2L sealed waste container
- **Water Tank**: 1L fresh water for flushing
- **Sanitizer**: Automatic spray after each use
- **Self-Cleaning**: UV sanitization cycle

**Toilet Specifications**:
| Specification | Value |
|---------------|-------|
| Deployment | Slide-out from rear |
| Privacy Screen | 3-sided pop-up |
| Waste Tank | 2L capacity |
| Water Tank | 1L capacity |
| Sanitizer | Auto-spray |
| Cleaning | UV sanitization |
| Deploy Time | 5 seconds |

**Voice Commands for Toilet**:
- "Deploy toilet" - Extend toilet from vehicle
- "Retract toilet" - Put toilet away
- "Privacy mode" - Deploy full privacy screens
- "Flush" - Flush and clean
- "Toilet status" - Check tank levels
- "Clean toilet" - Run cleaning cycle

**Toilet Configuration**:
```yaml
toilet_feature:
  enabled: true
  location: rear_compartment
  deployment: slide_out
  privacy_screens:
    enabled: true
    sides: 3
    height_cm: 120
  tanks:
    waste_capacity_liters: 2
    water_capacity_liters: 1
  sanitization:
    auto_sanitizer: true
    uv_cleaning: true
    cleaning_cycle_seconds: 30
  alerts:
    waste_tank_warning: 80      # Warn at 80% full
    water_tank_warning: 20      # Warn at 20% remaining
    sanitizer_warning: 10       # Warn at 10% remaining
```

---

**ATM Machine**:
A mini ATM built into the vehicle for withdrawing cash on the go!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class ATMState(Enum):
    STOWED = "stowed"              # Hidden inside vehicle
    ACTIVE = "active"              # Ready for transactions
    PROCESSING = "processing"      # Processing transaction
    DISPENSING = "dispensing"      # Dispensing cash
    ERROR = "error"                # Error state

class Currency(Enum):
    USD = "usd"                    # US Dollars
    EUR = "eur"                    # Euros
    GBP = "gbp"                    # British Pounds
    JPY = "jpy"                    # Japanese Yen

@dataclass
class ATMStatus:
    state: ATMState                # Current ATM state
    cash_available: dict[str, int] # Bills available by denomination
    total_cash: float              # Total cash in machine
    connected: bool                # Connected to bank network?
    last_transaction: str          # Last transaction timestamp

class IATMMachine(ABC):
    """Interface for ATM machine system"""

    @abstractmethod
    def activate(self) -> bool:
        """Activate ATM for transactions"""
        ...

    @abstractmethod
    def deactivate(self) -> bool:
        """Deactivate and stow ATM"""
        ...

    @abstractmethod
    def withdraw(self, amount: float, currency: Currency) -> bool:
        """Withdraw cash"""
        ...

    @abstractmethod
    def check_balance(self, card_data: str) -> float:
        """Check account balance"""
        ...

    @abstractmethod
    def get_status(self) -> ATMStatus:
        """Get ATM status"""
        ...

    @abstractmethod
    def refill_cash(self, denomination: str, count: int) -> None:
        """Refill cash into ATM"""
        ...

    @abstractmethod
    def print_receipt(self) -> bool:
        """Print transaction receipt"""
        ...
```

**ATM Hardware**:
- **Location**: Side panel with flip-out screen
- **Display**: 5-inch touchscreen
- **Card Reader**: Chip + contactless (NFC)
- **Cash Capacity**: $2,000 in mixed bills
- **Denominations**: $1, $5, $10, $20, $50, $100
- **Connectivity**: 4G/LTE for bank connection
- **Security**: Encrypted transactions, PIN pad
- **Receipt Printer**: Thermal mini printer

**ATM Specifications**:
| Specification | Value |
|---------------|-------|
| Cash Capacity | $2,000 |
| Denominations | $1, $5, $10, $20, $50, $100 |
| Display | 5-inch touchscreen |
| Card Types | Chip, swipe, NFC contactless |
| Connectivity | 4G/LTE encrypted |
| Receipt | Thermal printer |
| Max Withdrawal | $500 per transaction |

**Voice Commands for ATM**:
- "Open ATM" - Activate the ATM
- "Close ATM" - Deactivate and stow
- "ATM status" - Check cash levels
- "Refill ATM" - Open for refilling

**ATM Configuration**:
```yaml
atm_machine:
  enabled: true
  location: side_panel
  display:
    size_inches: 5
    type: touchscreen
  card_reader:
    chip: true
    swipe: true
    nfc: true
  cash_capacity:
    total_usd: 2000
    denominations:
      $1: 50
      $5: 50
      $10: 50
      $20: 50
      $50: 10
      $100: 5
  connectivity:
    type: 4g_lte
    encryption: aes_256
  limits:
    max_withdrawal: 500
    daily_limit: 1000
  receipt_printer:
    enabled: true
    paper_rolls: 2
```

---

**Vending Machine**:
A mini vending machine with snacks and drinks for on-the-go refreshments!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class ProductCategory(Enum):
    SNACKS = "snacks"              # Chips, cookies, etc.
    DRINKS = "drinks"              # Bottles and cans
    CANDY = "candy"                # Candy bars
    HEALTHY = "healthy"            # Granola, fruit snacks

class VendingState(Enum):
    STOWED = "stowed"              # Hidden inside vehicle
    ACTIVE = "active"              # Ready for purchases
    DISPENSING = "dispensing"      # Dispensing product
    RESTOCKING = "restocking"      # Being refilled

@dataclass
class Product:
    name: str                      # Product name
    category: ProductCategory      # Product category
    price: float                   # Price
    slot: str                      # Slot location (A1, B2, etc.)
    quantity: int                  # Items in stock

@dataclass
class VendingStatus:
    state: VendingState            # Current state
    products: list[Product]        # Available products
    total_items: int               # Total items in stock
    temperature_celsius: float     # Internal temperature
    revenue_today: float           # Today's sales

class IVendingMachine(ABC):
    """Interface for vending machine system"""

    @abstractmethod
    def activate(self) -> bool:
        """Activate vending machine"""
        ...

    @abstractmethod
    def deactivate(self) -> bool:
        """Deactivate and stow"""
        ...

    @abstractmethod
    def purchase(self, slot: str, payment_method: str) -> bool:
        """Purchase item from slot"""
        ...

    @abstractmethod
    def get_products(self) -> list[Product]:
        """Get available products"""
        ...

    @abstractmethod
    def get_status(self) -> VendingStatus:
        """Get vending machine status"""
        ...

    @abstractmethod
    def restock(self, slot: str, quantity: int) -> None:
        """Restock a product slot"""
        ...

    @abstractmethod
    def set_temperature(self, celsius: float) -> None:
        """Set cooling temperature"""
        ...
```

**Vending Machine Hardware**:
- **Location**: Side panel with glass display front
- **Display**: LED-lit glass door showing products
- **Slots**: 12 product slots (3 rows × 4 columns)
- **Capacity**: 24 items total (2 per slot)
- **Cooling**: Mini fridge for drinks (5°C)
- **Payment**: Contactless, card, or cash
- **Dispense**: Spiral coil mechanism

**Vending Machine Specifications**:
| Specification | Value |
|---------------|-------|
| Product Slots | 12 (3×4 grid) |
| Total Capacity | 24 items |
| Drink Cooling | 5°C mini fridge |
| Payment | NFC, card, cash |
| Display | LED-lit glass |
| Power | 50W cooling system |

**Available Products**:
| Slot | Product | Category | Price |
|------|---------|----------|-------|
| A1 | Chips | Snacks | $1.50 |
| A2 | Cookies | Snacks | $1.25 |
| A3 | Pretzels | Snacks | $1.25 |
| A4 | Crackers | Snacks | $1.00 |
| B1 | Water | Drinks | $1.00 |
| B2 | Soda | Drinks | $1.50 |
| B3 | Juice | Drinks | $2.00 |
| B4 | Energy Drink | Drinks | $2.50 |
| C1 | Candy Bar | Candy | $1.25 |
| C2 | Gummy Bears | Candy | $1.50 |
| C3 | Granola Bar | Healthy | $1.75 |
| C4 | Fruit Snacks | Healthy | $1.50 |

**Voice Commands for Vending Machine**:
- "Open vending machine" - Activate
- "Close vending machine" - Deactivate
- "What snacks do you have?" - List products
- "I want chips" / "Get me water" - Purchase
- "Vending status" - Check stock levels
- "Restock vending" - Open for restocking

**Vending Machine Configuration**:
```yaml
vending_machine:
  enabled: true
  location: side_panel
  display:
    type: led_glass
    size_inches: 8
  slots:
    rows: 3
    columns: 4
    capacity_per_slot: 2
  cooling:
    enabled: true
    target_celsius: 5
    for_rows: [B]  # Only drinks row
  payment:
    nfc: true
    card: true
    cash: true
  products:
    A1: { name: "Chips", price: 1.50, qty: 2 }
    A2: { name: "Cookies", price: 1.25, qty: 2 }
    A3: { name: "Pretzels", price: 1.25, qty: 2 }
    A4: { name: "Crackers", price: 1.00, qty: 2 }
    B1: { name: "Water", price: 1.00, qty: 2 }
    B2: { name: "Soda", price: 1.50, qty: 2 }
    B3: { name: "Juice", price: 2.00, qty: 2 }
    B4: { name: "Energy Drink", price: 2.50, qty: 2 }
    C1: { name: "Candy Bar", price: 1.25, qty: 2 }
    C2: { name: "Gummy Bears", price: 1.50, qty: 2 }
    C3: { name: "Granola Bar", price: 1.75, qty: 2 }
    C4: { name: "Fruit Snacks", price: 1.50, qty: 2 }
```

---

**Controller AI System**:
A smart AI assistant built into the controller that helps you drive, suggests actions, and makes controlling the MIKRO-CARRIER easier and more fun!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class AIPersonality(Enum):
    HELPFUL = "helpful"            # Friendly and informative
    SPORTY = "sporty"              # Energetic and competitive
    CALM = "calm"                  # Relaxed and patient
    FUNNY = "funny"                # Jokes and humor
    PROFESSIONAL = "professional"  # Serious and efficient

class AIMode(Enum):
    ASSISTANT = "assistant"        # Help with commands and suggestions
    COPILOT = "copilot"            # Active driving assistance
    AUTOPILOT = "autopilot"        # Full autonomous control
    TRAINING = "training"          # Teaching mode for new users
    GAME = "game"                  # Fun challenges and games

@dataclass
class AIStatus:
    active: bool                   # Is AI currently on?
    mode: AIMode                   # Current AI mode
    personality: AIPersonality     # Current personality
    suggestions_enabled: bool      # Giving suggestions?
    voice_enabled: bool            # Speaking responses?
    learning_user: bool            # Learning user preferences?

@dataclass
class AISuggestion:
    action: str                    # Suggested action
    reason: str                    # Why AI suggests this
    confidence: float              # How confident (0-1)
    priority: str                  # "low", "medium", "high"

class IControllerAI(ABC):
    """Interface for controller AI assistant"""

    @abstractmethod
    def activate(self) -> bool:
        """Turn on the AI assistant"""
        ...

    @abstractmethod
    def deactivate(self) -> bool:
        """Turn off the AI assistant"""
        ...

    @abstractmethod
    def set_mode(self, mode: AIMode) -> None:
        """Set AI operating mode"""
        ...

    @abstractmethod
    def set_personality(self, personality: AIPersonality) -> None:
        """Set AI personality style"""
        ...

    @abstractmethod
    def get_suggestion(self) -> AISuggestion:
        """Get AI's current suggestion"""
        ...

    @abstractmethod
    def ask_question(self, question: str) -> str:
        """Ask the AI a question"""
        ...

    @abstractmethod
    def execute_command(self, command: str) -> bool:
        """Tell AI to do something"""
        ...

    @abstractmethod
    def get_status(self) -> AIStatus:
        """Get AI status"""
        ...

    @abstractmethod
    def learn_preference(self, preference: str, value: any) -> None:
        """Teach AI your preferences"""
        ...
```

**Controller AI Hardware**:
- **Processor**: Dedicated AI chip on controller (NPU)
- **Display**: 4-inch touchscreen on controller
- **Microphone**: Built-in mic for voice commands
- **Speaker**: Small speaker for AI voice responses
- **Memory**: 4GB for AI model and user data
- **Connectivity**: Bluetooth + WiFi to vehicle

**Controller AI Specifications**:
| Specification | Value |
|---------------|-------|
| AI Chip | Dedicated NPU |
| Display | 4-inch touchscreen |
| Response Time | <500ms |
| Voice Recognition | Yes, always listening |
| Personalities | 5 different styles |
| Modes | 5 operating modes |
| Learning | Adapts to user over time |

**AI Modes Explained**:
| Mode | Description | Best For |
|------|-------------|----------|
| Assistant | Answers questions, gives tips | General use |
| Copilot | Helps steer, avoids obstacles | Learning to drive |
| Autopilot | Full self-driving | Hands-free operation |
| Training | Step-by-step tutorials | New users |
| Game | Challenges and competitions | Fun! |

**AI Personalities**:
| Personality | Voice Style | Example Response |
|-------------|-------------|------------------|
| Helpful | Friendly | "Great job! Try turning a bit earlier next time." |
| Sporty | Energetic | "Yeah! That drift was AWESOME! Let's go faster!" |
| Calm | Relaxed | "Nice and steady. You're doing well." |
| Funny | Humorous | "Whoa, that was close! My circuits almost fried!" |
| Professional | Serious | "Obstacle detected. Recommend evasive action." |

**AI Features**:
- **Driving Tips**: Suggests better driving techniques
- **Obstacle Warnings**: Alerts you to obstacles ahead
- **Battery Alerts**: Reminds you when battery is low
- **Feature Guide**: Explains how to use all features
- **Challenge Mode**: Sets fun driving challenges
- **Learning**: Remembers your preferences over time
- **Voice Chat**: Have conversations with the AI
- **Auto-Suggestions**: Proactively suggests actions

**Voice Commands for Controller AI**:
- "Hey AI" / "Hello" - Activate AI conversation
- "Help me" - Get assistance
- "What should I do?" - Get suggestion
- "Take over" / "Autopilot" - Enable autopilot mode
- "You drive" - Let AI control
- "I'll drive" - Take back control
- "Be funny" / "Be serious" - Change personality
- "Teach me" - Enter training mode
- "Challenge me" - Start a challenge
- "What's that?" - AI explains what it sees
- "Status report" - Full vehicle status
- "Good job" / "Bad suggestion" - Train the AI

**Controller AI Configuration**:
```yaml
controller_ai:
  enabled: true
  default_mode: assistant
  default_personality: helpful
  voice:
    enabled: true
    voice_type: friendly
    volume: 70
    speed: normal
  features:
    suggestions: true
    obstacle_warnings: true
    battery_alerts: true
    driving_tips: true
    learning: true
  autopilot:
    enabled: true
    max_speed_percent: 50      # Safety limit in autopilot
    obstacle_avoidance: true
    return_home: true          # Can auto-return if signal lost
  training:
    tutorials:
      - basic_driving
      - advanced_maneuvers
      - flight_basics
      - using_accessories
    adaptive_difficulty: true
  games:
    - time_trial
    - obstacle_course
    - precision_parking
    - follow_the_leader
    - treasure_hunt
  learning:
    remember_preferences: true
    adapt_suggestions: true
    track_improvement: true
```

**AI Conversation Examples**:
```
User: "Hey AI, what's my battery level?"
AI: "You've got 73% battery - plenty of juice for more fun!"

User: "How do I do a drift?"
AI: "For a drift, get some speed, then turn sharply while
     tapping the brake. Want me to show you in training mode?"

User: "Challenge me!"
AI: "Alright! Try to drive through all 5 gates in under
     30 seconds. Ready? 3... 2... 1... GO!"

User: "Be funny"
AI: "Switching to comedy mode! Warning: my jokes may cause
     excessive eye-rolling. Side effects include groaning."
```

---

**ICEE Machine**:
A frozen slushie dispenser for refreshing ICEE drinks on the go!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class ICEEFlavor(Enum):
    CHERRY = "cherry"              # Classic red cherry
    BLUE_RASPBERRY = "blue_raspberry"  # Blue raspberry
    COLA = "cola"                  # Coca-Cola flavor
    GRAPE = "grape"                # Purple grape
    LEMON_LIME = "lemon_lime"      # Sprite/7-Up style
    MYSTERY = "mystery"            # Mix of flavors!

class CupSize(Enum):
    SMALL = "small"                # 8 oz
    MEDIUM = "medium"              # 12 oz
    LARGE = "large"                # 16 oz

@dataclass
class ICEEStatus:
    machine_on: bool               # Is machine running?
    temperature_celsius: float     # Current temp (should be -2 to -4°C)
    flavors_available: dict[ICEEFlavor, int]  # Syrup levels per flavor
    ice_level_percent: int         # Ice/slush level
    cups_remaining: int            # Disposable cups left
    ready_to_dispense: bool        # Ready for serving?

class IICEEMachine(ABC):
    """Interface for ICEE slushie machine"""

    @abstractmethod
    def power_on(self) -> bool:
        """Turn on the ICEE machine"""
        ...

    @abstractmethod
    def power_off(self) -> bool:
        """Turn off the ICEE machine"""
        ...

    @abstractmethod
    def dispense(self, flavor: ICEEFlavor, size: CupSize) -> bool:
        """Dispense an ICEE drink"""
        ...

    @abstractmethod
    def mix_flavors(self, flavors: list[ICEEFlavor], size: CupSize) -> bool:
        """Dispense a mixed flavor ICEE"""
        ...

    @abstractmethod
    def get_status(self) -> ICEEStatus:
        """Get machine status"""
        ...

    @abstractmethod
    def refill_syrup(self, flavor: ICEEFlavor) -> None:
        """Refill a flavor syrup"""
        ...

    @abstractmethod
    def refill_cups(self, count: int) -> None:
        """Refill disposable cups"""
        ...

    @abstractmethod
    def clean_nozzles(self) -> None:
        """Run cleaning cycle"""
        ...
```

**ICEE Machine Hardware**:
- **Location**: Side panel with dispensing nozzle
- **Capacity**: 2 liters of slush per flavor
- **Flavors**: 6 flavor tanks
- **Freezing**: Compressor cooling system
- **Temperature**: Maintains -2 to -4°C
- **Cup Dispenser**: Holds 20 cups (small/medium/large)
- **Nozzles**: 2 dispensing nozzles for mixing

**ICEE Machine Specifications**:
| Specification | Value |
|---------------|-------|
| Flavor Tanks | 6 × 2L each |
| Total Capacity | 12 liters |
| Temperature | -2 to -4°C |
| Cup Sizes | 8oz, 12oz, 16oz |
| Cups Stored | 20 cups |
| Cooling | Mini compressor |
| Power | 60W cooling system |

**Available Flavors**:
| Flavor | Color | Description |
|--------|-------|-------------|
| Cherry | Red | Classic sweet cherry |
| Blue Raspberry | Blue | Tangy blue raspberry |
| Cola | Brown | Coca-Cola taste |
| Grape | Purple | Sweet grape |
| Lemon-Lime | Green | Citrus refreshing |
| Mystery | Rainbow | Mix surprise! |

**Voice Commands for ICEE Machine**:
- "ICEE please" / "Slushie" - Dispense default flavor
- "Cherry ICEE" / "Blue raspberry" - Specific flavor
- "Large ICEE" / "Small slushie" - Specify size
- "Mix cherry and blue" - Mixed flavor
- "ICEE status" - Check syrup and ice levels
- "How many cups left?" - Check cup count

**ICEE Machine Configuration**:
```yaml
icee_machine:
  enabled: true
  location: side_panel
  cooling:
    type: compressor
    target_temp_celsius: -3
    power_watts: 60
  tanks:
    capacity_liters: 2
    flavors:
      cherry: 100
      blue_raspberry: 100
      cola: 100
      grape: 100
      lemon_lime: 100
      mystery: 100
  cups:
    capacity: 20
    sizes: [small, medium, large]
  dispensing:
    nozzles: 2
    allow_mixing: true
    default_size: medium
    default_flavor: cherry
  maintenance:
    auto_clean_interval_hours: 24
    low_syrup_warning: 20
    low_cups_warning: 5
```

---

**TV Screen**:
A mounted TV screen above the ICEE machine for entertainment!

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class TVInput(Enum):
    STREAMING = "streaming"        # Netflix, YouTube, etc.
    VEHICLE_CAMERA = "vehicle_cam" # Live camera feeds
    GAMES = "games"                # Built-in games
    MEDIA_PLAYER = "media"         # USB/SD card media
    MIRROR = "mirror"              # Phone screen mirroring

class TVState(Enum):
    OFF = "off"                    # Screen off
    ON = "on"                      # Screen on
    STANDBY = "standby"            # Low power mode
    SCREENSAVER = "screensaver"    # Screensaver active

@dataclass
class TVStatus:
    state: TVState                 # Current state
    input_source: TVInput          # Current input
    volume: int                    # Volume level (0-100)
    brightness: int                # Brightness (0-100)
    current_content: str           # What's playing
    wifi_connected: bool           # Connected to internet?

class ITVScreen(ABC):
    """Interface for TV screen system"""

    @abstractmethod
    def power_on(self) -> bool:
        """Turn on the TV"""
        ...

    @abstractmethod
    def power_off(self) -> bool:
        """Turn off the TV"""
        ...

    @abstractmethod
    def set_input(self, source: TVInput) -> None:
        """Change input source"""
        ...

    @abstractmethod
    def set_volume(self, level: int) -> None:
        """Set volume (0-100)"""
        ...

    @abstractmethod
    def set_brightness(self, level: int) -> None:
        """Set brightness (0-100)"""
        ...

    @abstractmethod
    def play_content(self, content: str) -> bool:
        """Play specific content"""
        ...

    @abstractmethod
    def get_status(self) -> TVStatus:
        """Get TV status"""
        ...

    @abstractmethod
    def mirror_phone(self) -> bool:
        """Enable phone screen mirroring"""
        ...

    @abstractmethod
    def show_camera(self, camera_id: str) -> None:
        """Show vehicle camera feed"""
        ...
```

**TV Screen Hardware**:
- **Location**: Mounted above ICEE machine
- **Display**: 7-inch IPS LCD
- **Resolution**: 1280 × 800 (HD)
- **Brightness**: 500 nits (outdoor visible)
- **Speakers**: Built-in 2W stereo speakers
- **Connectivity**: WiFi, Bluetooth, HDMI-in
- **Storage**: 16GB for downloaded content

**TV Screen Specifications**:
| Specification | Value |
|---------------|-------|
| Screen Size | 7 inches |
| Resolution | 1280 × 800 HD |
| Brightness | 500 nits |
| Speakers | 2 × 1W stereo |
| WiFi | Yes (streaming) |
| Bluetooth | Yes (audio out) |
| Storage | 16GB |
| Inputs | Streaming, Camera, Games, USB |

**TV Features**:
| Feature | Description |
|---------|-------------|
| Streaming | YouTube, Netflix, Disney+ |
| Camera View | Live feeds from all vehicle cameras |
| Games | Built-in casual games |
| Media Player | Play videos from USB/SD |
| Phone Mirror | Cast phone screen to TV |
| Picture-in-Picture | Watch TV while seeing camera |

**Voice Commands for TV**:
- "TV on" / "TV off" - Power control
- "Watch YouTube" / "Play Netflix" - Start streaming
- "Show front camera" - Vehicle camera feed
- "Play games" - Open game menu
- "Volume up" / "Volume down" - Adjust volume
- "Brightness up" / "Brightness down" - Adjust brightness
- "Mirror my phone" - Screen mirroring
- "What's on TV?" - Current content info

**TV Screen Configuration**:
```yaml
tv_screen:
  enabled: true
  location: above_icee_machine
  display:
    size_inches: 7
    resolution: "1280x800"
    brightness_nits: 500
    auto_brightness: true
  audio:
    speakers: stereo
    power_watts: 2
    bluetooth_out: true
  connectivity:
    wifi: true
    bluetooth: true
    hdmi_in: true
    usb: true
    sd_card: true
  streaming:
    apps:
      - youtube
      - netflix
      - disney_plus
      - spotify
    require_wifi: true
  camera_feeds:
    - front_camera
    - rear_camera
    - clone_cameras
    - drone_view
  games:
    - racing
    - puzzle
    - trivia
    - arcade
  storage_gb: 16
  power:
    auto_off_minutes: 30
    screensaver_minutes: 5
```

**Entertainment Setup**:
```
    ┌─────────────────┐
    │    📺 TV        │  ← 7" HD Screen
    │   (7 inch)      │
    ├─────────────────┤
    │  🧊 ICEE        │  ← Slushie Dispenser
    │   Machine       │
    │  🔴🔵🟣🟢🟤    │  ← 6 Flavors
    └─────────────────┘
```

---

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

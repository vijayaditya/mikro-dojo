# RC Car Multi-View Learning System - Architecture Design Document

## Executive Summary

This document specifies a reality-native robot learning system that enables RC-scale vehicles to acquire complex motor skills through observation, extended offline processing, and iterative physical refinement. The system leverages external arena observation combined with onboard sensing to create a dense supervision signal unavailable to traditional robot learning approaches.

**Key Differentiators:**
- 1000x iteration advantage through safe, bounded arena enabling failure-based learning
- Exogenous observation providing third-person perspective for skill assessment
- Offline learning paradigm removing real-time computational constraints
- VLM-guided continuous evaluation eliminating hand-crafted reward engineering
- Reality-native training avoiding sim-to-real transfer losses

## 1. System Architecture Overview

### 1.1 Conceptual Framework

The system operates on a **five-phase learning cycle**:

1. **Demonstration Capture**: Human demonstrates skill via RC controller while system records from multiple perspectives
2. **Offline Processing**: Hours to days of computational processing extracting behavioral primitives and learning correspondences
3. **Arena Experimentation**: Autonomous execution attempts with systematic variation exploration
4. **Iterative Refinement**: VLM-guided assessment drives policy improvement through reinforcement learning
5. **Skill Deployment**: Optimized policy deployed to edge hardware for real-time execution

### 1.2 Architectural Principles

**Separation of Concerns:**
- **Perception Layer**: Handles all sensor data acquisition, synchronization, and low-level processing
- **Learning Layer**: Manages offline training, correspondence learning, and policy optimization
- **Evaluation Layer**: Provides continuous skill assessment through VLM integration
- **Execution Layer**: Ensures real-time control with safety guarantees
- **Orchestration Layer**: Coordinates between layers and manages learning curriculum

**Scalability Considerations:**
- Support for 1-10 simultaneous RC cars in shared arena
- Extensible to 100+ unique skills
- Cloud-native learning pipeline supporting distributed training
- Modular sensor architecture allowing configuration changes without redesign

## 2. Hardware Architecture Specification

### 2.1 Arena Infrastructure Requirements

**Physical Dimensions:**
- Minimum viable: 3m × 3m × 2m (height)
- Recommended: 5m × 5m × 2.5m for complex maneuvers
- Maximum practical: 10m × 10m × 3m for multi-agent scenarios

**Boundary System:**
- Material: Energy-absorbing foam (minimum 10cm thickness)
- Configuration: Modular panels allowing arena reconfiguration
- Height: Minimum 50cm above maximum jump height
- Safety factor: 2x maximum kinetic energy absorption

**Floor Specifications:**
- Base: Level surface with <5mm variation over 3m span
- Friction coefficient: 0.6-1.2 (variable via overlay materials)
- Markings: High-contrast patterns for visual odometry
- Modularity: Swappable surface materials for terrain variation

### 2.2 External Observation Network

**Camera Array Topology:**

**Overhead Coverage:**
- Quantity: 4 cameras minimum for complete coverage
- Placement: Grid pattern with 20% overlap between fields of view
- Resolution: 4K (3840×2160) minimum
- Frame rate: 30fps synchronized
- Lens: Wide-angle (90-120° FOV) with minimal distortion

**Perimeter Coverage:**
- Quantity: 8 cameras (2 per wall)
- Placement: 45° down-angle at 1.5m height
- Purpose: Capture vehicle dynamics from side perspectives
- Resolution: 1080p minimum
- Frame rate: 60fps for dynamic maneuvers

**Specialized Cameras:**
- High-speed units: 2 cameras at 240fps for rapid motion capture
- Depth cameras: Optional RGBD units for 3D reconstruction
- Trigger: Motion-activated or skill-specific activation

**Synchronization Requirements:**
- Maximum frame offset: <1ms across all cameras
- Method: Hardware trigger preferred, NTP-based software sync acceptable
- Timestamp precision: Microsecond resolution
- Calibration: Extrinsic parameters updated weekly

### 2.3 RC Vehicle Platform Specifications

**Chassis Requirements:**
- Scale: 1/10 or 1/8 for optimal sensor payload
- Drive: 4WD with independent motor control
- Suspension: Adjustable for different terrains
- Maximum speed: Software-limited to 20mph for safety
- Payload capacity: Minimum 1.5kg for sensors and compute

**Onboard Compute Architecture:**

**Primary Processing Unit:**
- Minimum: NVIDIA Jetson Orin Nano (40 TOPS)
- Recommended: NVIDIA Jetson AGX Orin (275 TOPS)
- Memory: 8GB minimum, 16GB preferred
- Storage: 128GB NVMe for local data buffering

**Real-time Controller:**
- Microcontroller: ARM Cortex-M7 or equivalent
- Loop rate: 1000Hz for motor control
- Interfaces: PWM, I2C, SPI, CAN
- Redundancy: Watchdog timer with hardware reset

**Sensor Suite Requirements:**

**Visual Perception:**
- Forward stereo: Baseline 6-10cm, global shutter preferred
- Rear camera: Wide-angle for situation awareness
- Resolution: 720p minimum at 30fps
- Latency: <50ms from capture to processing

**Inertial Measurement:**
- 9-DOF IMU minimum (gyro, accel, mag)
- Update rate: 200Hz minimum
- Noise characteristics: <0.01°/s/√Hz gyro, <100μg/√Hz accelerometer
- Temperature compensation: Required for drift mitigation

**Proprioceptive Sensing:**
- Wheel encoders: 2048 CPR minimum resolution
- Motor current: Per-motor sensing at 100Hz
- Battery monitoring: Voltage, current, temperature
- Steering angle: Absolute position sensor if Ackermann steering

## 3. Data Architecture

### 3.1 Data Flow Hierarchy

**Level 1: Raw Sensor Streams**
- Arena cameras: 8 × 4K × 30fps = ~1.5GB/s
- Onboard cameras: 2 × 720p × 30fps = ~100MB/s
- IMU data: 200Hz × 72 bytes = ~14KB/s
- Encoders: 1000Hz × 16 bytes = ~16KB/s
- Total bandwidth: ~1.6GB/s raw data

**Level 2: Synchronized Datasets**
- Temporal alignment: ±1ms accuracy
- Spatial registration: All cameras in unified coordinate frame
- Format: Hierarchical structure with metadata
- Compression: H.265 for video, lossless for sensor data
- Storage requirement: ~500GB per hour of demonstrations

**Level 3: Processed Training Data**
- Extracted features: Visual embeddings, trajectory segments
- Correspondence pairs: Ego-exo matched frames
- Augmented variations: Synthetic viewpoints, temporal shifts
- Format: Optimized for parallel training pipeline
- Size: ~50GB per skill after processing

**Level 4: Learned Representations**
- Policy networks: 10-100MB per skill
- Correspondence models: ~500MB shared across skills
- Value functions: ~50MB per skill
- Deployment models: 5-20MB after optimization

### 3.2 Data Collection Protocol

**Pre-Collection Setup:**
1. Arena configuration documentation (obstacle positions, surface type)
2. Sensor calibration verification (maximum 7 days old)
3. Time synchronization (<1ms drift across all systems)
4. Lighting consistency check (±20% variation acceptable)

**Demonstration Recording:**
- Human demonstrator uses standard RC controller
- Minimum 10 successful demonstrations per skill
- Variety requirements: Different speeds, approaches, styles
- Failure examples: 5-10 controlled failures for boundary learning
- Duration: 30 seconds to 5 minutes per demonstration

**Quality Assurance:**
- Real-time visualization of all camera feeds
- Synchronization verification every 10 seconds
- Data integrity checks (no dropped frames, sensor dropouts)
- Immediate replay capability for demonstrator review

**Post-Collection Processing:**
- Automated quality scoring of demonstrations
- Outlier detection and removal
- Temporal segmentation into skill primitives
- Metadata annotation (weather, time, demonstrator ID)

### 3.3 Storage and Retention Strategy

**Tier 1: Hot Storage (NVMe SSD)**
- Current training data (last 7 days)
- Active model checkpoints
- Real-time buffer for ongoing collections
- Capacity: 2TB minimum

**Tier 2: Warm Storage (SATA SSD)**
- Recent demonstrations (last 30 days)
- Validated model versions
- Preprocessed datasets
- Capacity: 10TB recommended

**Tier 3: Cold Storage (HDD/Cloud)**
- Historical demonstrations for longitudinal studies
- Archived model checkpoints
- Raw sensor data for reprocessing
- Capacity: Unlimited cloud storage

**Data Lifecycle:**
- Raw data: Retain 30 days then archive
- Processed datasets: Retain indefinitely
- Model checkpoints: Keep best 3 per skill version
- Execution logs: 90-day retention

## 4. Learning System Architecture

### 4.1 Multi-Stage Learning Pipeline

**Stage 1: Behavioral Primitive Extraction**

*Purpose*: Decompose demonstrations into reusable motor primitives

*Input Requirements*:
- Minimum 10 demonstrations of target skill
- Multi-view synchronized video
- Control inputs from demonstrator

*Processing Specifications*:
- Unsupervised segmentation using change point detection
- Clustering similar motion patterns across demonstrations
- Primitive duration: 0.5-3 seconds typically
- Output: Library of 20-50 primitive motions

*Success Criteria*:
- 95% of demonstration reconstructable from primitives
- Primitives generalize across different demonstrations
- Temporal consistency across multiple executions

**Stage 2: Ego-Exo Correspondence Learning**

*Purpose*: Learn mapping between external observation and robot's perspective

*Architecture Requirements*:
- Dual-encoder design with shared latent space
- Contrastive learning with hard negative mining
- Temporal consistency constraints
- View synthesis capability for missing perspectives

*Training Specifications*:
- Batch size: 128 frame pairs minimum
- Negative samples: 10:1 ratio to positive
- Temperature parameter: 0.05-0.1 for contrastive loss
- Training duration: 10,000-50,000 iterations

*Validation Metrics*:
- Correspondence accuracy: >90% for synchronized frames
- Temporal consistency: <5% drift over 10-second sequences
- View synthesis quality: SSIM >0.85

**Stage 3: Dynamics Model Learning**

*Purpose*: Understand vehicle physics and action-outcome relationships

*Model Specifications*:
- Architecture: Recurrent or transformer-based
- History window: 1-2 seconds of past states
- Prediction horizon: 0.5-1 second future
- Update rate: 50Hz minimum

*Training Requirements*:
- State representation: Position, velocity, acceleration, orientation
- Action space: Throttle, steering, brake
- Loss function: Weighted MSE with emphasis on contact events
- Regularization: Physics-based constraints (energy conservation)

*Validation Criteria*:
- Prediction accuracy: <10cm position error at 1 second
- Contact prediction: >85% accuracy for collisions
- Generalization: Performance maintained on unseen terrains

**Stage 4: Policy Learning via Imitation**

*Purpose*: Learn initial control policy from demonstrations

*Architecture Design*:
- Input: Multi-modal state (visual + proprioceptive)
- Processing: Attention-based fusion of modalities
- Output: Continuous actions with uncertainty estimates
- Capacity: 1-10M parameters

*Training Protocol*:
- Dataset: Filtered high-quality demonstrations
- Augmentation: Noise injection, viewpoint variation
- Curriculum: Easy to hard trajectory segments
- Iterations: 100K-500K gradient steps

*Performance Targets*:
- Demonstration matching: >70% trajectory similarity
- Smoothness: Action variation <20% between timesteps
- Safety: Zero boundary violations in validation

**Stage 5: Reinforcement Learning Refinement**

*Purpose*: Improve beyond demonstrations through trial and error

*Algorithm Selection*:
- On-policy: PPO for stable learning
- Off-policy: SAC for sample efficiency
- Model-based: Optional for planning

*Reward Architecture*:
- Dense rewards: Progress toward goal, smoothness
- Sparse rewards: Skill completion
- VLM evaluation: Aesthetic and style assessment
- Safety penalties: Boundary violations, dangerous states

*Training Specifications*:
- Episodes: 10,000-100,000 per skill
- Exploration strategy: Adaptive noise with decay
- Update frequency: Every 2048 steps (PPO)
- Parallelization: 4-8 simultaneous rollout workers

*Convergence Criteria*:
- Success rate: >90% for core skills
- Reward plateau: <1% improvement over 1000 episodes
- Safety: <0.1% dangerous state occurrences

### 4.2 VLM Integration Architecture

**Evaluation Pipeline Design:**

*Frame Selection Strategy*:
- Keyframe extraction at 2Hz for efficiency
- Event-triggered capture for critical moments
- Multi-view aggregation for comprehensive assessment
- Temporal window: 5-30 seconds per evaluation

*Prompt Engineering Framework*:
- Skill-specific evaluation criteria
- Structured output format (JSON)
- Chain-of-thought reasoning for explanations
- Comparative assessment against ideal execution

*Evaluation Metrics Hierarchy*:
1. Task completion (binary)
2. Execution quality (0-10 scale)
3. Safety compliance (violations count)
4. Style matching (similarity score)
5. Efficiency metrics (time, energy)

*Integration Points*:
- Real-time: Coarse evaluation during execution
- Post-episode: Detailed analysis for learning
- Batch: Comparative ranking of multiple attempts
- Human-in-loop: Override for edge cases

### 4.3 Continual Learning Architecture

**Experience Replay System:**
- Buffer capacity: 1M transitions minimum
- Prioritization: Based on TD-error and novelty
- Forgetting: FIFO with importance weighting
- Compression: State abstraction for old experiences

**Skill Composition Framework:**
- Hierarchical skill representation
- Primitive reuse across complex skills
- Automatic curriculum generation
- Transfer learning between related skills

**Meta-Learning Components:**
- Fast adaptation to new skills (10-shot learning)
- Hyperparameter optimization
- Architecture search for skill-specific models
- Automatic feature selection

## 5. Deployment Architecture

### 5.1 Model Optimization Pipeline

**Compression Stages:**

1. **Pruning**: Remove 50-90% of weights while maintaining performance
2. **Quantization**: INT8 or FP16 precision with calibration
3. **Knowledge Distillation**: Train smaller student model
4. **Architecture Optimization**: Replace complex ops with efficient alternatives

**Target Specifications:**
- Model size: <20MB for edge deployment
- Inference latency: <20ms on target hardware
- Memory footprint: <500MB runtime
- Power consumption: <5W average

**Validation Requirements:**
- Performance degradation: <5% from original
- Safety preservation: 100% of safety constraints maintained
- Robustness testing: Performance under noise/perturbations
- Hardware-in-loop verification: Test on actual platform

### 5.2 Runtime Architecture

**Hierarchical Control Structure:**

**Level 1: Skill Executor (50Hz)**
- Input: Current state, skill selection
- Processing: Policy inference
- Output: Desired actions
- Latency budget: 20ms

**Level 2: Safety Monitor (100Hz)**
- Input: Proposed actions, current state
- Processing: Constraint checking
- Output: Safe actions
- Latency budget: 10ms

**Level 3: Motor Controller (1000Hz)**
- Input: Safe actions
- Processing: PID control, PWM generation
- Output: Motor commands
- Latency budget: 1ms

**State Management:**
- Circular buffers for sensor history
- Predictive state estimation during sensor delays
- Graceful degradation on sensor failure
- Emergency stop on critical failures

**Communication Architecture:**
- Inter-process: Shared memory for low latency
- Remote monitoring: MQTT for telemetry
- Model updates: Secure OTA with rollback
- Debugging: Comprehensive logging with rotation

### 5.3 Safety System Design

**Multi-Layer Safety Architecture:**

**Hardware Layer:**
- Independent emergency stop circuit
- Current limiting on motors
- Mechanical limits on steering
- Battery protection circuits

**Firmware Layer:**
- Watchdog timers
- Sensor sanity checks
- Rate limiters
- Fail-safe defaults

**Software Layer:**
- State space constraints
- Action space limits
- Predictive collision detection
- Stability monitoring

**Learning Layer:**
- Safe exploration strategies
- Constraint-aware training
- Adversarial testing
- Formal verification where possible

**Operational Constraints:**
- Maximum velocity: 5m/s in learning mode
- Acceleration limits: 2g maximum
- Arena boundary: 50cm safety margin
- Human proximity: Auto-stop within 1m

## 6. System Integration Specifications

### 6.1 Software Architecture

**Microservice Decomposition:**

1. **Data Acquisition Service**
   - Responsibility: Sensor data collection and synchronization
   - Interface: gRPC for streaming data
   - Scaling: Horizontal for multiple robots
   - State: Stateless with external storage

2. **Learning Orchestrator**
   - Responsibility: Coordinate training pipeline
   - Interface: REST API for job submission
   - Scaling: Vertical for compute intensity
   - State: Persistent job queue

3. **Model Registry**
   - Responsibility: Version control for models
   - Interface: REST for CRUD operations
   - Scaling: Replicated for availability
   - State: Persistent with backup

4. **Evaluation Service**
   - Responsibility: VLM integration and scoring
   - Interface: Async message queue
   - Scaling: Horizontal with GPU pool
   - State: Stateless with result cache

5. **Deployment Service**
   - Responsibility: Model optimization and distribution
   - Interface: REST with webhook notifications
   - Scaling: On-demand workers
   - State: Build artifacts in object storage

### 6.2 Development Workflow

**Skill Development Lifecycle:**

1. **Definition Phase**
   - Skill specification document
   - Success criteria definition
   - Safety constraint identification
   - Resource requirement estimation

2. **Collection Phase**
   - Demonstration protocol creation
   - Data quality validation
   - Annotation and metadata
   - Version control integration

3. **Training Phase**
   - Hyperparameter selection
   - Distributed training orchestration
   - Continuous validation
   - Experiment tracking

4. **Validation Phase**
   - Simulation testing
   - Arena testing protocol
   - Safety verification
   - Performance benchmarking

5. **Deployment Phase**
   - Model optimization
   - Edge deployment
   - A/B testing capability
   - Rollback procedures

### 6.3 Monitoring and Observability

**Metrics Collection:**

**System Metrics:**
- Compute utilization (GPU, CPU, memory)
- Network throughput and latency
- Storage IOPS and capacity
- Power consumption

**Application Metrics:**
- Training loss curves
- Inference latency distribution
- Model accuracy over time
- Skill success rates

**Business Metrics:**
- Skills learned per week
- Demonstration efficiency
- System availability
- Cost per skill

**Alerting Thresholds:**
- Critical: Safety violations, system failures
- Warning: Performance degradation, resource exhaustion
- Info: Training milestones, successful deployments

**Visualization Requirements:**
- Real-time dashboards for operations
- Training progress visualization
- Arena activity heatmaps
- Skill execution replay capability

## 7. Scalability and Performance Requirements

### 7.1 Performance Targets

**Training Performance:**
- Behavioral cloning: <1 hour per skill
- Correspondence learning: <4 hours
- RL refinement: <24 hours to convergence
- End-to-end: 48 hours from demonstration to deployment

**Inference Performance:**
- Edge latency: <20ms at 95th percentile
- Throughput: 50Hz sustained
- Cold start: <2 seconds
- Model loading: <500ms

**Data Pipeline Performance:**
- Ingestion: 2GB/s sustained
- Processing: 100GB/hour throughput
- Storage: 10TB capacity minimum
- Query: <100ms for recent data

### 7.2 Scalability Dimensions

**Robot Scaling:**
- Support 1-10 robots in single arena
- Independent learning pipelines
- Shared model improvements
- Collision avoidance for multi-agent

**Skill Scaling:**
- 100+ unique skills
- Hierarchical organization
- Composite skill construction
- Transfer learning optimization

**User Scaling:**
- Multi-tenant architecture
- Resource isolation
- Priority scheduling
- Fair-share allocation

**Compute Scaling:**
- Auto-scaling training workers
- Spot instance utilization
- Distributed training support
- Edge compute optimization

### 7.3 Resource Requirements

**Minimum Viable System:**
- 1 GPU (RTX 3090 or better)
- 64GB system RAM
- 2TB SSD storage
- Gigabit networking

**Recommended Production:**
- 4 GPUs (A100 or H100)
- 256GB system RAM
- 10TB NVMe array
- 10Gb networking

**Cloud Alternative:**
- GPU instances for training
- Serverless for orchestration
- Object storage for datasets
- CDN for model distribution

## 8. Success Criteria and KPIs

### 8.1 Technical Success Metrics

**Learning Efficiency:**
- Demonstrations required: <20 per skill
- Training time: <48 hours to deployment
- Success rate: >90% for learned skills
- Generalization: >70% on variations

**System Reliability:**
- Uptime: >99.5% for core services
- Data loss: <0.01% of collections
- Model corruption: Zero incidents
- Recovery time: <1 hour from failure

**Safety Metrics:**
- Collision rate: <1 per 1000 episodes
- Boundary violations: <0.1%
- Emergency stops: <1 per day
- Human injuries: Zero tolerance

### 8.2 Operational KPIs

**Efficiency Indicators:**
- Cost per skill: <$100 in compute
- Human effort: <2 hours per skill
- Iteration velocity: 100+ attempts/day
- Resource utilization: >70% GPU usage

**Quality Measurements:**
- VLM scores: >8/10 average
- Human evaluation: >85% approval
- Skill retention: >95% after 30 days
- Transfer success: >60% to new skills

**Scale Achievements:**
- Skills library: 50+ in first year
- Robot fleet: 5+ simultaneous
- Users supported: 10+ researchers
- Datasets: 1000+ hours collected

## 9. Risk Analysis and Mitigation

### 9.1 Technical Risks

**Risk: Sim-to-Real Gap in Dynamics**
- Mitigation: Reality-native training emphasis
- Fallback: Hybrid sim+real approach
- Monitoring: Continuous validation metrics

**Risk: VLM Evaluation Inconsistency**
- Mitigation: Ensemble of evaluators
- Fallback: Human validation sampling
- Monitoring: Inter-rater reliability tracking

**Risk: Catastrophic Forgetting**
- Mitigation: Experience replay, regularization
- Fallback: Separate models per skill
- Monitoring: Performance regression tests

### 9.2 Operational Risks

**Risk: Data Collection Bottleneck**
- Mitigation: Parallel collection, automation
- Fallback: Synthetic data augmentation
- Monitoring: Collection rate metrics

**Risk: Hardware Failures**
- Mitigation: Redundant sensors, spare robots
- Fallback: Simulation-based training
- Monitoring: Component health tracking

**Risk: Scalability Limits**
- Mitigation: Distributed architecture design
- Fallback: Vertical scaling option
- Monitoring: Resource utilization trends

### 9.3 Safety Risks

**Risk: Runaway Robot**
- Mitigation: Physical arena boundaries
- Fallback: Emergency stop systems
- Monitoring: Continuous state validation

**Risk: Human Injury**
- Mitigation: Speed limits, proximity detection
- Fallback: Remote operation only
- Monitoring: Safety incident logging

**Risk: Property Damage**
- Mitigation: Energy-absorbing boundaries
- Fallback: Reduced power operation
- Monitoring: Impact force measurement

## 10. Future Extensibility

### 10.1 Planned Enhancements

**Phase 2 Capabilities:**
- Multi-robot coordination
- Adversarial training
- Sim2real transfer learning
- Custom skill programming interface

**Phase 3 Capabilities:**
- Outdoor operation
- Variable scale robots
- Human-robot interaction
- Real-time learning

**Phase 4 Capabilities:**
- Full autonomy
- Self-improvement
- Novel skill discovery
- Cross-platform transfer

### 10.2 Research Opportunities

**Algorithm Development:**
- Few-shot skill learning
- Compositional reasoning
- Causal understanding
- Uncertainty quantification

**System Improvements:**
- Edge-cloud hybrid training
- Federated learning
- Neural architecture search
- Automated curriculum learning

**Application Domains:**
- Education platform
- Research testbed
- Competition framework
- Entertainment system

## Appendix A: Skill Taxonomy

### Basic Skills (Difficulty 1-3)
- Forward/Reverse driving
- Point turns
- Lane following
- Speed control
- Smooth stopping

### Intermediate Skills (Difficulty 4-6)
- Parallel parking
- Three-point turns
- Slalom navigation
- Controlled slides
- Ramp climbing

### Advanced Skills (Difficulty 7-9)
- Drifting corners
- Donuts
- J-turns
- Handbrake turns
- Jump control

### Expert Skills (Difficulty 10)
- Barrel rolls
- Wall rides
- Precision jumps
- Stunt combinations
- Recovery from flips

## Appendix B: Data Formats

### Sensor Data Schema
- Timestamp: int64, nanoseconds since epoch
- Frame ID: Sequential counter per sensor
- Data: Binary blob or structured format
- Metadata: JSON with calibration, settings
- Checksum: CRC32 for integrity

### Training Data Organization
- Sessions: Top-level grouping by date
- Skills: Subdirectories per skill type
- Demonstrations: Individual recording folders
- Processed: Derived data in parallel structure
- Models: Version-controlled checkpoints

### Model Artifact Structure
- Architecture: JSON description
- Weights: Binary format (ONNX preferred)
- Metadata: Training configuration, metrics
- Validation: Test results and benchmarks
- Deployment: Optimized versions per platform

## Appendix C: Interface Specifications

### REST API Endpoints
- `/skills` - CRUD operations on skills
- `/demonstrations` - Upload/query demos
- `/training` - Submit/monitor training jobs
- `/models` - Model registry operations
- `/deployment` - Deploy to robot fleet
- `/evaluation` - Request VLM assessment

### Message Queue Topics
- `sensor.data.*` - Raw sensor streams
- `training.events` - Training progress
- `evaluation.results` - VLM outputs
- `robot.telemetry` - Runtime metrics
- `system.alerts` - Operational events

### gRPC Services
- `DataCollection` - Streaming sensor data
- `ModelServing` - Inference requests
- `Orchestration` - Job management
- `Monitoring` - Metrics collection

---

*This design document serves as the authoritative reference for implementing the RC car multi-view learning system. All implementation decisions should align with these specifications while maintaining flexibility for iterative improvements based on empirical results.*
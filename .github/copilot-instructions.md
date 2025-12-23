# BP-SLAM Copilot Instructions

## Project Overview

**BP-SLAM** is a Python implementation of a Belief Propagation-based multipath-assisted Simultaneous Localization and Mapping (SLAM) algorithm. It uses particle filtering and belief propagation to estimate an agent's trajectory and anchor point locations in environments with multipath propagation and clutter. The project is a Python conversion of the original MATLAB implementation.

**Key Paper**: Meyer & Leitinger et al. - "Belief Propagation based Multipath-assisted SLAM"

## Architecture Overview

### Core Components

```
bp_slam/
©À©¤©¤ core/              # SLAM algorithm core
©¦   ©À©¤©¤ slam.py        # Main algorithm: bp_based_mint_slam()
©¦   ©À©¤©¤ anchors.py     # Anchor management (initialization, prediction, generation)
©¦   ©¸©¤©¤ association.py # Data association using Belief Propagation
©À©¤©¤ utils/             # Utility functions
©¦   ©À©¤©¤ belief_propagation.py  # BP message passing computations
©¦   ©À©¤©¤ motion_model.py        # State transition matrices & prediction
©¦   ©À©¤©¤ measurements.py        # Measurement generation (clean & cluttered)
©¦   ©À©¤©¤ sampling.py            # Particle sampling & resampling
©¦   ©¸©¤©¤ distance.py            # Distance calculations
©¸©¤©¤ visualization/     # Plotting and visualization
```

### Data Flow

1. **Input**: MATLAB `.mat` files containing virtual anchor (VA) data and true trajectory
2. **Measurement Generation**: `measurements.generate_measurements()` ¡ú `generate_cluttered_measurements()` (adds clutter & missed detections)
3. **SLAM Main Loop**: `bp_based_mint_slam()` iterates through time steps:
   - **Prediction**: `perform_prediction()` updates agent particles based on motion model
   - **Anchor Management**: `predict_anchors()`, `generate_new_anchors()`, `delete_unreliable_va()`
   - **Data Association**: `data_association_bp()` uses belief propagation to match measurements to anchors
   - **Update**: Particle filtering with systematic resampling
4. **Output**: Estimated trajectory, anchor positions, and posterior particles

### State Representation

- **Agent State**: 4D vector `[x, y, vx, vy]` (position + velocity)
- **Anchor State**: 2D position with existence probability
- **Particles**: Collections of weighted samples representing posterior distributions

## Key Algorithms & Concepts

### Belief Propagation (Data Association)

Located in `association.py::data_association_bp()` and `belief_propagation.py`:

- Uses iterative message passing to assign measurements to anchors
- Handles uncertainty in both measurement and anchor existence
- Converges based on message change threshold (default check every iteration, threshold ~0.01)
- Returns association probabilities matrix for measurement-anchor pairs

**Key function**: `get_input_bp()` combines anchor existence probability with measurement likelihoods:
```
message_out = existence * message_target_is_present + (1 - existence) * message_target_is_absent
```

### Particle Filtering

- **Particles**: Represent posterior distributions of agent state and anchor positions
- **Resampling**: Uses systematic resampling (`resample_systematic()`) when effective sample size drops below threshold
- **Weights**: Updated based on measurement likelihoods; normalized after each update
- **Motion Model**: Linear constant-velocity with Gaussian acceleration noise

### Anchor Management

Three types of anchors:
1. **Physical Anchors (PA)**: Known anchor positions initialized from `data_va['positions']`
2. **Virtual/Geometric Anchors (VA)**: Generated via multipath propagation
3. **New Anchors**: Generated dynamically using `generate_new_anchors()`

Key functions:
- `init_anchors()`: Initialize particle sets for known anchors
- `predict_anchors()`: Perform prediction step for all anchors
- `generate_new_anchors()`: Sample new anchor hypotheses from measurements
- `delete_unreliable_va()`: Remove anchors below existence probability threshold

## Critical Parameters (testbed.py)

```python
# Motion & Measurement
parameters['lengthStep'] = 0.03           # Distance per time step (m)
parameters['scanTime'] = 1                # Time between measurements (s)
parameters['measurementVariance'] = 0.01  # Range measurement noise
parameters['drivingNoiseVariance'] = ...  # Velocity process noise

# Detection & Clutter
parameters['detectionProbability'] = 0.95 # Probability of detecting anchor
parameters['clutterIntensity'] = ¦Ë_c      # False alarm rate per unit area
parameters['survivalProbability'] = 0.999 # Anchor survival probability

# Particle Filtering
parameters['numParticles'] = 100000       # Number of particles (large!)
parameters['unreliabilityThreshold'] = 1e-4  # Min existence prob to keep

# Region & Birth Process
parameters['regionOfInterestSize'] = 30   # Area side length (m)
parameters['birthIntensity'] = ¦Ë_b        # New anchor birth rate
parameters['undetectedAnchorsIntensity'] = ¦Ë_u  # Undetected anchor intensity
```

## Common Development Tasks

### Running Tests

```bash
# Full algorithm (900 steps, 100k particles - slow, ~2-3 min)
python testbed.py

# Quick test (100 steps, 10k particles - ~10-30s)
python testbed_quick.py

# Visualization of results
python visualize_results.py
```

### Adding/Modifying Algorithms

1. **Motion model changes**: Edit `motion_model.py::get_transition_matrices()` for state transition, adjust noise in `perform_prediction()`
2. **Measurement model changes**: Modify `measurements.py::generate_measurements()` (range calculation), adjust `measurementVariance` parameter
3. **Belief propagation tuning**: In `association.py::data_association_bp()`, adjust `check_convergence` interval and convergence `threshold`

### Debugging Tips

- **Particle degeneracy**: Check `num_particles` in parameters and effective sample size calculations in `slam.py`
- **Poor trajectory estimates**: Verify `drivingNoiseVariance` (too large ¡ú noisy, too small ¡ú filter diverges)
- **Anchor mismanagement**: Check `unreliabilityThreshold` and `birthIntensity` balance
- **Slow execution**: Reduce `numParticles` or `maxSteps` for testing; numpy operations should be vectorized

## Project-Specific Conventions

1. **Array Shapes**: Consistently use `(dimension, num_samples)` for 2D data (e.g., particles: `(4, numParticles)`)
2. **Indexing**: Python 0-based (MATLAB was 1-based in original); watch for off-by-one errors in conversions
3. **Data Structures**: Mix of numpy arrays and dictionaries (e.g., anchor particles have `{'x', 'w', 'posteriorExistence'}`)
4. **Measurement Format**: Always `(2, num_measurements)` where row 0 = range, row 1 = variance
5. **Time Steps**: Iterate with `for step in range(num_steps)`, particles always shape `(4, num_particles)` or `(2, num_particles)`

## Integration Points & Dependencies

- **scipy.io**: Load MATLAB `.mat` files (required for input data)
- **numpy**: All numerical computations; expect heavy broadcasting
- **matplotlib**: Visualization only; not required for core algorithm
- **Particle Filter Loop** (slam.py): Tightly coupled with `anchors.py` and `association.py`¡ªchanges to one affect convergence
- **Virtual Anchor Generation**: Depends on `measurements.py` likelihoods; check consistency of variance parameters

## Files to Know

| File | Purpose |
|------|---------|
| `bp_slam/core/slam.py` | Main SLAM loop; start here for algorithm flow |
| `testbed.py` / `testbed_quick.py` | Entry points; parameter definitions |
| `bp_slam/core/association.py` | Belief propagation core; critical for understanding data association |
| `bp_slam/utils/belief_propagation.py` | BP message utilities |
| `bp_slam/utils/sampling.py` | Resampling routines |
| `data_set/location{0,1,2,3}/data.json` | Test datasets with real measurements |


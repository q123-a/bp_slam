# -*- coding: utf-8 -*-
"""
Graph optimization backend using GTSAM

Core modules:
- FeatureManager: Feature lifecycle manager (tentative -> active)
- GraphBackend: GTSAM graph optimization backend
"""

import numpy as np

try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    print("Warning: GTSAM not installed. Graph backend will not work.")

from ..utils.geometry_utils import triangulate_feature, compute_distance


class FeatureManager:
    """
    Feature lifecycle manager
    
    Manages features from "tentative" to "active" state:
    - tentative: Newly observed feature, accumulating observations
    - active: Successfully triangulated, added to graph optimization
    
    Keyframe decimation: Only saves observations 0.2m apart to improve
    triangulation matrix condition number.
    """
    
    def __init__(self, min_observations=5, min_baseline=0.8, rmse_threshold=0.5):
        """
        Initialize feature manager
        
        Args:
            min_observations: Minimum keyframes for activation (default 5)
                             With 0.2m spacing, 5 keyframes = 1.0m travel
            min_baseline: Minimum baseline distance in meters (default 0.8)
            rmse_threshold: Triangulation RMSE threshold (default 0.5)
        """
        # Buffer for tentative features: {fid: {'poses': [], 'ranges': [], 'probs': []}}
        self.tentative_buffer = {}
        
        # Set of active feature IDs
        self.active_features = set()
        
        # Initialized positions for active features (for GTSAM re-initialization)
        self.active_feature_positions = {}  # {fid: [x, y]}
        
        # Miss counts for pruning
        self.miss_counts = {}
        
        # Parameters
        self.min_observations = min_observations
        self.min_baseline = min_baseline
        self.rmse_threshold = rmse_threshold
        
        # Global feature ID counter
        self.next_fid = 0
        
        # Mapping from (sensor_id, anchor_idx) to global feature ID
        self.anchor_to_fid = {}
        
    def get_or_create_fid(self, sensor_id, anchor_idx):
        """
        Get or create global feature ID
        
        Args:
            sensor_id: Sensor ID
            anchor_idx: Anchor index in sensor
            
        Returns:
            global_fid: Global feature ID
        """
        key = (sensor_id, anchor_idx)
        if key not in self.anchor_to_fid:
            self.anchor_to_fid[key] = self.next_fid
            self.next_fid += 1
        return self.anchor_to_fid[key]
    
    def process(self, fid, current_pose, rng, prob=1.0, anchor_pos_hint=None, pf_particles=None):
        """
        Process new observation, return feature status
        
        Uses PF particle convergence to decide when to initialize GTSAM.
        "Ç°ÆÚ¿¿ PF Ö¸Â·£¬ºóÆÚ¿¿ GTSAM ¾«ÐÞ"
        
        Args:
            fid: Global feature ID
            current_pose: Current agent position [x, y]
            rng: Range measurement
            prob: Association probability from BP
            anchor_pos_hint: Optional hint for anchor position (from BP particle mean)
            pf_particles: Optional PF particles for this feature, shape (2, num_particles)
            
        Returns:
            (status, init_pos):
                status: "UPDATE" - Active feature, update directly
                        "INIT" - Newly activated, return initial position
                        "WAIT" - Tentative, keep accumulating
                init_pos: Initial position (only valid for INIT)
        """
        # Reset miss count
        if fid in self.miss_counts:
            self.miss_counts[fid] = 0
        
        # Case 1: Already active
        if fid in self.active_features:
            return "UPDATE", None
        
        # Case 2: New or tentative -> add to buffer
        if fid not in self.tentative_buffer:
            self.tentative_buffer[fid] = {
                'poses': [],
                'ranges': [],
                'probs': [],
                'anchor_hints': [],
                'pf_particles_history': []  # Track particle convergence
            }
        
        buffer = self.tentative_buffer[fid]
        
        # --- Keyframe decimation strategy ---
        # Save data when moved at least 0.1m from last saved pose
        # With 0.03m/step, this requires ~3-4 steps between keyframes
        should_save = False
        if len(buffer['poses']) == 0:
            should_save = True
        else:
            last_pose = buffer['poses'][-1]
            if np.linalg.norm(current_pose - last_pose) > 0.1:  # 0.1m spacing
                should_save = True
        
        if should_save:
            buffer['poses'].append(current_pose.copy())
            buffer['ranges'].append(rng)
            buffer['probs'].append(prob)
            if anchor_pos_hint is not None:
                buffer['anchor_hints'].append(anchor_pos_hint.copy())
        
        # --- PF Particle Convergence Check ---
        # This is the key "handoff" logic: only inject into GTSAM when PF has converged
        pf_converged = False
        pf_uncertainty = float('inf')
        pf_mean = None
        
        if pf_particles is not None and pf_particles.shape[1] > 0:
            # Calculate particle distribution statistics
            # particles shape: (2, num_particles)
            std_dev = np.std(pf_particles, axis=1)  # [std_x, std_y]
            pf_uncertainty = np.linalg.norm(std_dev)
            pf_mean = np.mean(pf_particles, axis=1)
            
            # Track convergence history
            buffer['pf_particles_history'].append(pf_uncertainty)
            
            # Debug: print convergence status periodically
            if len(buffer['pf_particles_history']) % 10 == 0:
                print(f"[Debug] Feature {fid}: keyframes={len(buffer['poses'])}, "
                      f"pf_uncertainty={pf_uncertainty:.3f}m, converged={pf_uncertainty < 0.5}")
            
            # Convergence criteria:
            # - Uncertainty < 0.5m means PF has resolved the ambiguity
            # - Uncertainty > 1.0m means PF is still "choosing sides"
            if pf_uncertainty < 0.5:
                pf_converged = True
        
        # --- FAST PATH: For extremely confident features (PA), skip keyframe requirements ---
        # Physical anchors have known positions, PF knows them from the start
        if pf_converged and pf_uncertainty < 0.1 and prob > 0.7:
            print(f"[PA Init] Feature {fid} initialized as known anchor "
                  f"(uncertainty={pf_uncertainty:.3f}m)")
            self.active_features.add(fid)
            self.active_feature_positions[fid] = pf_mean.copy()
            self.miss_counts[fid] = 0
            if fid in self.tentative_buffer:
                del self.tentative_buffer[fid]
            return "INIT", pf_mean
        
        # Case 3: Try initialization (for VA - virtual anchors)
        # Strategy: 共线期用PF，非共线期用三角化
        if len(buffer['ranges']) >= self.min_observations:
            poses_arr = np.array(buffer['poses'])
            baseline = np.linalg.norm(poses_arr[-1] - poses_arr[0]) if len(poses_arr) > 1 else 0
            
            # Check geometric condition: is the trajectory collinear?
            is_collinear = True
            min_eig = 0.0
            if len(poses_arr) >= 3:
                # Compute covariance eigenvalues to detect collinearity
                data = poses_arr[:, :2] - np.mean(poses_arr[:, :2], axis=0)
                cov = np.dot(data.T, data) / len(data)
                eig_vals = np.linalg.eigvals(cov)
                min_eig = min(eig_vals)
                # If min eigenvalue > threshold, trajectory has lateral movement (not collinear)
                is_collinear = (min_eig < 2e-3)
            
            # Debug log for geometry check (every 5 observations)
            if len(buffer['poses']) % 5 == 0:
                print(f"[Geometry] Feature {fid}: obs={len(buffer['poses'])}, "
                      f"baseline={baseline:.2f}m, min_eig={min_eig:.4f}, collinear={is_collinear}, "
                      f"pf_unc={pf_uncertainty:.3f}m")
            
            if is_collinear:
                # === 共线期：使用放宽的PF收敛条件 ===
                # 虽然几何上有镜像歧义，但如果PF通过其他观测（PA等）已选定一侧并收敛
                # 可以使用较宽松的阈值激活
                # 条件: PF不确定度 < 1.0m（比PA的0.5m宽松，允许一定残余歧义）
                if pf_uncertainty < 1.0 and baseline >= 0.5 and prob > 0.5:
                    print(f"[PF-Collinear] Feature {fid} initialized via PF "
                          f"(collinear but converged, uncertainty={pf_uncertainty:.3f}m, baseline={baseline:.2f}m)")
                    self.active_features.add(fid)
                    self.active_feature_positions[fid] = pf_mean.copy()
                    self.miss_counts[fid] = 0
                    del self.tentative_buffer[fid]
                    return "INIT", pf_mean
            else:
                # === 非共线期：歧义已打破，可以激活 ===
                # 方法1: PF收敛（转弯后双峰变单峰）
                # 此时PF应该能收敛到较低不确定度
                if pf_converged and baseline >= 0.3 and prob > 0.5:
                    print(f"[PF-NonCollinear] Feature {fid} initialized via PF "
                          f"(non-collinear, uncertainty={pf_uncertainty:.3f}m, baseline={baseline:.2f}m)")
                    self.active_features.add(fid)
                    self.active_feature_positions[fid] = pf_mean.copy()
                    self.miss_counts[fid] = 0
                    del self.tentative_buffer[fid]
                    return "INIT", pf_mean
                
                # 方法2: 三角化（几何上可唯一确定）
                is_valid, est_pos = triangulate_feature(
                    buffer['poses'], 
                    buffer['ranges'],
                    min_observations=self.min_observations,
                    rmse_threshold=self.rmse_threshold
                )
                
                if is_valid:
                    print(f"[Triangulation] Feature {fid} initialized via triangulation "
                          f"(non-collinear, min_eig={min_eig:.4f}, baseline={baseline:.2f}m)")
                    self.active_features.add(fid)
                    self.active_feature_positions[fid] = est_pos.copy()
                    self.miss_counts[fid] = 0
                    del self.tentative_buffer[fid]
                    return "INIT", est_pos
        
        return "WAIT", None
    
    def update_miss_counts(self, observed_fids, max_miss=10):
        """
        Update miss counts for unobserved features
        
        Args:
            observed_fids: Set of observed feature IDs this frame

            max_miss: Maximum allowed consecutive misses
            
        Returns:
            pruned_fids: List of pruned feature IDs
        """
        pruned = []
        for fid in list(self.active_features):
            if fid not in observed_fids:
                self.miss_counts[fid] = self.miss_counts.get(fid, 0) + 1
                if self.miss_counts[fid] > max_miss:
                    pruned.append(fid)
        
        for fid in pruned:
            self.active_features.discard(fid)
            if fid in self.miss_counts:
                del self.miss_counts[fid]
        
        return pruned
    
    def get_buffer_info(self, fid):
        """
        Get tentative feature buffer info (for back-filling)
        
        Args:
            fid: Feature ID
            
        Returns:
            Buffer dict or None
        """
        return self.tentative_buffer.get(fid, None)


class GraphBackend:
    """
    GTSAM-based graph optimization backend
    
    Uses ISAM2 for incremental optimization, converts BP association
    probabilities to observation weights.
    
    Supports two-phase operation:
    - Phase 1 (PF): Use particle filter estimates, only add motion factors to GTSAM
    - Phase 2 (GTSAM): Switch to graph optimization when features converge
    """
    
    # Phase constants
    PHASE_PF = 0      # Pure PF mode, GTSAM only tracks motion
    PHASE_GTSAM = 1   # GTSAM optimization mode
    
    def __init__(self, parameters=None):
        """
        Initialize graph optimization backend
        
        Args:
            parameters: Algorithm parameters dict
        """
        if not GTSAM_AVAILABLE:
            raise RuntimeError("GTSAM is not installed. Please install it first.")
        
        self.parameters = parameters or {}
        
        # Initialize ISAM2 optimizer
        isam_params = gtsam.ISAM2Params()
        isam_params.setRelinearizeThreshold(0.1)
        isam_params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(isam_params)
        
        # Factor graph containers
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        
        # State management
        self.frame_count = 0
        self.last_pose = np.array([0., 0.])
        self.last_velocity = np.array([0., 0.])
        self.last_time = 0.0
        
        # ===== Two-phase operation =====
        self.current_phase = self.PHASE_PF
        self.phase_switch_frame = -1  # Frame when switched to GTSAM
        self.converged_features_count = 0  # Track how many features have converged
        self.min_converged_for_gtsam = 2  # Need at least 2 converged features to switch
        
        # Feature manager - adjusted for slow-moving agent
        self.feat_mgr = FeatureManager(
            min_observations=5,   # Need more observations for stable init
            min_baseline=0.05,    # Agent moves 0.03m per step
            rmse_threshold=1.0    # More tolerant
        )
        
        # Initialized landmarks (avoid duplicate additions)
        self.initialized_landmarks = set()
        
        # Noise models - use larger noise to be more robust
        self.PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3]))
        self.MOTION_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05]))
        
        # Base measurement noise

        self.base_sigma = np.sqrt(self.parameters.get('measurementVariance', 0.01))
    
    def initialize(self, initial_pose, initial_velocity=None):
        """
        Initialize backend (first frame)
        
        Args:
            initial_pose: Initial position [x, y]
            initial_velocity: Initial velocity [vx, vy]
        """
        self.last_pose = np.array(initial_pose[:2])
        self.last_velocity = np.array(initial_velocity[:2]) if initial_velocity is not None else np.zeros(2)
        
        # Add first position node and prior
        curr_key = gtsam.symbol('x', 0)
        self.graph.add(gtsam.PriorFactorPoint2(
            curr_key, 
            gtsam.Point2(self.last_pose[0], self.last_pose[1]),
            self.PRIOR_NOISE
        ))
        self.initial_estimates.insert(curr_key, gtsam.Point2(self.last_pose[0], self.last_pose[1]))
        
        # Initial optimization
        self.isam.update(self.graph, self.initial_estimates)
        self.graph.resize(0)
        self.initial_estimates.clear()
        
        self.frame_count = 1
    
    def update(self, timestamp, sensor_data, predicted_pose=None, anchor_particles=None, pf_position=None):
        """
        Core update function with two-phase operation
        
        Phase 1 (PF): Only accumulate observations, return PF position
        Phase 2 (GTSAM): Full graph optimization
        
        Args:
            timestamp: Current timestamp
            sensor_data: List of sensor observations
            predicted_pose: Predicted agent position (from motion model)
            anchor_particles: Dict of PF particles for each anchor
            pf_position: Current PF estimated position (used in Phase 1)
            
        Returns:
            (estimated_pose, estimated_velocity, phase): Position, velocity, current phase
        """
        dt = timestamp - self.last_time if self.frame_count > 0 else 0.1
        if dt < 1e-6:
            dt = 0.1
        
        # Determine which position to use
        if predicted_pose is not None:
            pred_pos = np.array(predicted_pose[:2])
        else:
            pred_pos = self.last_pose + self.last_velocity * dt
        
        if anchor_particles is None:
            anchor_particles = {}
        
        # --- Process observations and track convergence ---
        observed_fids = set()
        newly_converged = 0
        
        for obs in sensor_data:
            sensor_id = obs['sensor_id']
            anchor_idx = obs['anchor_idx']
            rng = obs['range']
            prob = obs.get('prob', 1.0)
            existence = obs.get('existence', 1.0)
            anchor_pos = obs.get('position', None)
            
            if prob < 0.3 or existence < 0.3:
                continue
            
            fid = self.feat_mgr.get_or_create_fid(sensor_id, anchor_idx)
            observed_fids.add(fid)
            
            pf_particles = anchor_particles.get((sensor_id, anchor_idx), None)
            
            # Feature lifecycle management
            status, init_pos = self.feat_mgr.process(
                fid, pred_pos, rng, prob, 
                anchor_pos_hint=anchor_pos,
                pf_particles=pf_particles
            )
            
            # Track newly initialized features
            if status == "INIT":
                newly_converged += 1
        
        # Update miss counts
        self.feat_mgr.update_miss_counts(observed_fids)
        
        # Count total active features
        self.converged_features_count = len(self.feat_mgr.active_features)
        
        # --- Phase switching logic ---
        if self.current_phase == self.PHASE_PF:
            # Check if we should switch to GTSAM
            # Conditions for switching:
            # 1. Have enough converged features (at least 2)
            # 2. Have accumulated enough frames for baseline (at least 30 frames = ~0.9m travel)
            # 3. This ensures GTSAM has enough observations to constrain the solution
            min_frames_before_switch = 30
            
            can_switch = (
                self.converged_features_count >= self.min_converged_for_gtsam and
                self.frame_count >= min_frames_before_switch
            )
            
            if can_switch:
                print(f"\n{'='*50}")
                print(f"[Phase Switch] PF -> GTSAM at frame {self.frame_count}")
                print(f"  Converged features: {self.converged_features_count}")
                print(f"{'='*50}\n")
                self.current_phase = self.PHASE_GTSAM
                self.phase_switch_frame = self.frame_count
                
                # Re-initialize GTSAM with current PF position
                self._reinitialize_gtsam_from_pf(pf_position if pf_position is not None else pred_pos)
                
                # For the switch frame, return current PF position (GTSAM will start from next frame)
                result_pose = pf_position if pf_position is not None else pred_pos
                self.last_pose = np.array(result_pose[:2])
                self.last_time = timestamp
                self.frame_count += 1
                return self.last_pose, self.last_velocity, self.current_phase
        
        # --- Execute based on current phase ---
        if self.current_phase == self.PHASE_PF:
            # Phase 1: Use pure PF position
            if pf_position is not None:
                result_pose = np.array(pf_position[:2])
            else:
                result_pose = pred_pos
            
            # Update velocity from PF
            if dt > 1e-6:
                new_vel = (result_pose - self.last_pose) / dt
                vel_norm = np.linalg.norm(new_vel)
                if vel_norm > 2.0:
                    new_vel = new_vel / vel_norm * 2.0
                self.last_velocity = 0.7 * self.last_velocity + 0.3 * new_vel
            
            self.last_pose = result_pose
            self.last_time = timestamp
            self.frame_count += 1
            
            return result_pose, self.last_velocity, self.current_phase
        
        else:
            # Phase 2: Full GTSAM optimization
            return self._gtsam_update(timestamp, sensor_data, pred_pos, anchor_particles)
    
    def _reinitialize_gtsam_from_pf(self, current_pose):
        """
        Re-initialize GTSAM graph when switching from PF phase
        """
        # Clear old graph
        self.graph.resize(0)
        self.initial_estimates.clear()
        self.isam = gtsam.ISAM2(gtsam.ISAM2Params())
        
        # Add current position as starting point
        curr_key = gtsam.symbol('x', self.frame_count)
        self.graph.add(gtsam.PriorFactorPoint2(
            curr_key, 
            gtsam.Point2(current_pose[0], current_pose[1]),
            self.PRIOR_NOISE
        ))
        self.initial_estimates.insert(curr_key, gtsam.Point2(current_pose[0], current_pose[1]))
        
        # Add all converged features with their saved positions
        for fid in self.feat_mgr.active_features:
            if fid in self.feat_mgr.active_feature_positions:
                pos = self.feat_mgr.active_feature_positions[fid]
                land_key = gtsam.symbol('l', fid)
                self.initial_estimates.insert(land_key, gtsam.Point2(pos[0], pos[1]))
                weak_prior = gtsam.noiseModel.Isotropic.Sigma(2, 100.0)
                self.graph.add(gtsam.PriorFactorPoint2(
                    land_key,
                    gtsam.Point2(pos[0], pos[1]),
                    weak_prior
                ))
                self.initialized_landmarks.add(fid)
                print(f"  [GTSAM Init] Feature {fid} at ({pos[0]:.2f}, {pos[1]:.2f})")
        
        # Initial optimization
        try:
            self.isam.update(self.graph, self.initial_estimates)
            self.graph.resize(0)
            self.initial_estimates.clear()
        except Exception as e:
            print(f"[Warning] GTSAM re-initialization failed: {e}")
    
    def _gtsam_update(self, timestamp, sensor_data, pred_pos, anchor_particles):
        """
        Full GTSAM update (Phase 2)
        """
        dt = timestamp - self.last_time if self.frame_count > 0 else 0.1
        if dt < 1e-6:
            dt = 0.1
            
        # --- A. State node definition ---
        curr_key = gtsam.symbol('x', self.frame_count)
        prev_key = gtsam.symbol('x', self.frame_count - 1)
        
        # --- B. Motion model ---
        expected_disp = pred_pos - self.last_pose
        
        self.graph.add(gtsam.BetweenFactorPoint2(
            prev_key, curr_key,
            gtsam.Point2(expected_disp[0], expected_disp[1]),
            self.MOTION_NOISE
        ))
        self.initial_estimates.insert(curr_key, gtsam.Point2(pred_pos[0], pred_pos[1]))
        
        # --- C. Measurement processing ---
        for obs in sensor_data:
            sensor_id = obs['sensor_id']
            anchor_idx = obs['anchor_idx']
            rng = obs['range']
            prob = obs.get('prob', 1.0)
            existence = obs.get('existence', 1.0)
            anchor_pos = obs.get('position', None)
            
            if prob < 0.3 or existence < 0.3:
                continue
            
            fid = self.feat_mgr.get_or_create_fid(sensor_id, anchor_idx)
            land_key = gtsam.symbol('l', fid)
            
            # Build noise model
            sigma = max(0.1, self.base_sigma / (prob * existence + 0.001))
            noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                gtsam.noiseModel.Isotropic.Sigma(1, sigma)
            )
            
            if fid in self.initialized_landmarks and prob > 0.5:
                self.graph.add(gtsam.RangeFactor2(curr_key, land_key, rng, noise))
            elif fid not in self.initialized_landmarks and anchor_pos is not None and prob > 0.7:
                # Late initialization for newly converged features
                self.initial_estimates.insert(land_key, gtsam.Point2(anchor_pos[0], anchor_pos[1]))
                weak_prior = gtsam.noiseModel.Isotropic.Sigma(2, 100.0)
                self.graph.add(gtsam.PriorFactorPoint2(
                    land_key, gtsam.Point2(anchor_pos[0], anchor_pos[1]), weak_prior
                ))
                self.initialized_landmarks.add(fid)
                self.graph.add(gtsam.RangeFactor2(curr_key, land_key, rng, noise))
        
        # --- D. Execute optimization ---
        try:
            self.isam.update(self.graph, self.initial_estimates)
            self.graph.resize(0)
            self.initial_estimates.clear()
            
            result = self.isam.calculateEstimate()
            curr_pt = result.atPoint2(curr_key)
            if isinstance(curr_pt, np.ndarray):
                curr_pose_np = curr_pt
            else:
                curr_pose_np = np.array([curr_pt.x(), curr_pt.y()])
            
            # Position jump protection
            jump = np.linalg.norm(curr_pose_np - pred_pos)
            if jump > 1.0:
                print(f"[GraphBackend] Position jump {jump:.2f}m, using prediction")
                curr_pose_np = pred_pos
            
            # Velocity update
            if dt > 1e-6:
                new_vel = (curr_pose_np - self.last_pose) / dt
                vel_norm = np.linalg.norm(new_vel)
                if vel_norm > 2.0:
                    new_vel = new_vel / vel_norm * 2.0
                self.last_velocity = 0.7 * self.last_velocity + 0.3 * new_vel
            
            self.last_pose = curr_pose_np
            
        except Exception as e:
            print(f"GTSAM optimization failed: {e}")
            self.last_pose = pred_pos
            self.graph.resize(0)
            self.initial_estimates.clear()
        
        self.last_time = timestamp
        self.frame_count += 1
        
        return self.last_pose, self.last_velocity, self.current_phase
    
    def get_landmark_estimates(self):
        """
        Get all active feature position estimates
        
        Returns:
            dict: {fid: [x, y]}
        """
        landmarks = {}
        try:
            result = self.isam.calculateEstimate()
            for fid in self.initialized_landmarks:
                land_key = gtsam.symbol('l', fid)
                if result.exists(land_key):
                    pt = result.atPoint2(land_key)
                    if isinstance(pt, np.ndarray):
                        landmarks[fid] = pt
                    else:
                        landmarks[fid] = np.array([pt.x(), pt.y()])
        except Exception:
            pass
        return landmarks
    
    def get_trajectory(self):
        """
        Get complete trajectory estimate
        
        Returns:
            np.array: shape (2, num_frames)
        """
        trajectory = []
        try:
            result = self.isam.calculateEstimate()
            for i in range(self.frame_count):
                key = gtsam.symbol('x', i)
                if result.exists(key):
                    pt = result.atPoint2(key)
                    if isinstance(pt, np.ndarray):
                        trajectory.append(pt.tolist())
                    else:
                        trajectory.append([pt.x(), pt.y()])
        except Exception:
            pass
        
        return np.array(trajectory).T if trajectory else np.zeros((2, 0))


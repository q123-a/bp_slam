"""
bp_slam/core/gnn_trainer_v3.py
Dual-Head GNN Trainer V3 - Using Explicit Factor Node JointDualHeadGNNV3 Model

Core Improvements:
1. Uses JointDualHeadGNNV3 model (explicit factor node architecture)
2. Two-stage message passing: variable -> factor -> variable
3. Retains all V2 training strategies (fuzzy logic, dual-teacher self-supervised, adaptive weighting)
4. Stronger expressiveness, suitable for complex association scenarios

Explicit Factor Node Advantages:
- Factor nodes can learn complex "measurement-anchor" association patterns
- More aligned with factor graph mathematical structure (Belief Propagation)
- Can explicitly model different types of constraints (geometric, physical, temporal)

Interface Compatibility:
- Fully compatible with V2 trainer interface
- Input/output format remains consistent
- Can be directly replaced
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from .gnn_model_v3_explicit_factor import JointDualHeadGNNV3


class JointDualHeadTrainerV3:
    """
    Dual-Head Joint Trainer V3: Quality Head + Association Head Self-Supervised Training
    Using Explicit Factor Node JointDualHeadGNNV3 Model

    Core Innovations:
    1. Explicit Factor Node Architecture:
       - Measurement Node -> Factor Node -> Anchor Node
       - Two-stage message passing, stronger expressiveness

    2. Physics Teacher: Supervises quality head
       - Uses RSS intrinsic consistency
       - Three-interval fuzzy logic training strategy

    3. Geometry Teacher: Supervises association head
       - Hungarian algorithm global optimal matching
       - Only trains on high-quality measurements

    4. Adaptive Weight Balancing:
       - Dynamically adjusts loss weights based on training stage
       - Prevents overfitting of either head
    """

    def __init__(self, device='cuda', lr=1e-3, hidden_dim=64, checkpoint_path=None, seed=42,
                 use_ema=True, ema_decay=0.999, use_lr_scheduler=True,
                 use_temporal_gru=False, use_layer_gru=False,
                 quality_threshold=0.5, assoc_threshold=3.0,
                 quality_weight=1.0, assoc_weight=2.0,
                 adaptive_weighting=True, debug_interval=50,
                 aggregation='softmax', gamma=3.0, edge_mode='concat',
                 skip_connections=None):
        """
        Initialize Dual-Head Trainer V3

        Args:
            device: Device ('cuda' or 'cpu')
            lr: Initial learning rate
            hidden_dim: Hidden layer dimension
            checkpoint_path: Weight file path
            seed: Random seed
            use_ema: Whether to use EMA
            ema_decay: EMA decay rate
            use_lr_scheduler: Whether to use learning rate scheduler
            use_temporal_gru: Whether to use cross-frame GRU
            use_layer_gru: Whether to use intra-layer GRU
            quality_threshold: Quality judgment threshold (0.5)
            assoc_threshold: Association gating threshold (3.0)
            quality_weight: Quality loss initial weight (1.0)
            assoc_weight: Association loss initial weight (2.0)
            adaptive_weighting: Whether to use adaptive weight balancing
            debug_interval: Debug info print interval (steps)
            aggregation: Aggregation method ('softmax', 'max', 'mean')
            gamma: Softmax temperature parameter (default 3.0)
            edge_mode: Edge feature mode ('diff', 'concat')
            skip_connections: Skip connection dict {source_layer: target_layer}
        """
        self.device = device
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.use_temporal_gru = use_temporal_gru
        self.use_layer_gru = use_layer_gru
        self.quality_threshold = quality_threshold
        self.assoc_threshold = assoc_threshold
        self.quality_weight = quality_weight
        self.assoc_weight = assoc_weight
        self.adaptive_weighting = adaptive_weighting
        self.debug_interval = debug_interval
        self.aggregation = aggregation
        self.gamma = gamma
        self.edge_mode = edge_mode
        self.skip_connections = skip_connections

        # Set random seed
        if seed is not None:
            self._set_seed(seed)

        # Initialize model (using explicit factor node V3 model)
        self.model = JointDualHeadGNNV3(
            input_dim=5,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_temporal_gru=use_temporal_gru,
            use_layer_gru=use_layer_gru,
            aggregation=aggregation,
            gamma=gamma,
            edge_mode=edge_mode,
            skip_connections=skip_connections
        ).to(device)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # EMA model (optional)
        if use_ema:
            self.ema_model = JointDualHeadGNNV3(
                input_dim=5,
                hidden_dim=hidden_dim,
                num_layers=2,
                use_temporal_gru=use_temporal_gru,
                use_layer_gru=use_layer_gru,
                aggregation=aggregation,
                gamma=gamma,
                edge_mode=edge_mode,
                skip_connections=skip_connections
            ).to(device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
            print(f"EMA enabled (decay={ema_decay})")
        else:
            self.ema_model = None

        # Learning rate scheduler
        if use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=lr * 0.01
            )
            print(f"Learning rate scheduler enabled (CosineAnnealing)")
        else:
            self.scheduler = None

        # Training state
        self.step_count = 0
        self.loss_history = []
        self.quality_loss_history = []
        self.assoc_loss_history = []

        # GRU hidden state (maintained internally by trainer)
        # Changed to dict to support multiple sensors: {sensor_id: hidden_state}
        self.hidden_states = {}

        # Adaptive weight history
        self.weight_history = []

        # Load weights (if provided)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print(f"Dual-Head GNN Trainer V3 initialized")
        print(f"  - Model: JointDualHeadGNNV3 (explicit factor nodes)")
        print(f"  - Aggregation: {aggregation} (gamma={gamma})")
        print(f"  - Edge mode: {edge_mode}")
        print(f"  - Quality threshold: {quality_threshold}")
        print(f"  - Association threshold: {assoc_threshold}")
        print(f"  - Loss weights: quality={quality_weight}, assoc={assoc_weight}")
        print(f"  - Adaptive weighting: {adaptive_weighting}")
        print(f"  - Architecture: measurement->factor->anchor (two-stage message passing)")

    def _set_seed(self, seed):
        """Set random seed"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def reset_hidden_state(self, sensor_id=None):
        """
        Reset GRU hidden state

        Args:
            sensor_id: If specified, only reset this sensor's state.
                      If None, reset all sensors' states.

        Should be called when:
        1. Starting a new trajectory sequence
        2. Detecting trajectory interruption
        3. Measurement count or anchor count changes significantly
        """
        if sensor_id is None:
            # Reset all sensors
            self.hidden_states = {}
        else:
            # Reset specific sensor
            if sensor_id in self.hidden_states:
                del self.hidden_states[sensor_id]

    def step(self, hybrid_tensor, measurements, predicted_measurements, predicted_variances, num_iterations=1, sensor_id=0):
        """
        Execute one training/inference step

        Improvements:
        1. GRU state maintained internally by trainer (no external passing needed)
        2. Returns dustbin_probs instead of quality_scores (more aligned with SLAM interface)
        3. Gradient clipping prevents BPTT from being too long
        4. Supports multiple iteration training
        5. Supports multiple sensors with independent GRU states

        Input:
            hybrid_tensor: (1, M, K+1, 5) hybrid features
            measurements: (3, M) measurement data [distance, variance, amplitude]
            predicted_measurements: (K,) predicted distances
            predicted_variances: (K,) predicted variances
            num_iterations: Number of iterations per time step (default 1)
            sensor_id: Sensor identifier for independent GRU state management (default 0)

        Output:
            assoc_probs: (M, K) association probabilities
            dustbin_probs: (M,) clutter probabilities [0, 1]
            final_scale: (M,) variance inflation scale (currently all ones)
            loss: Scalar loss value
        """
        self.step_count += 1
        hybrid_tensor = hybrid_tensor.to(self.device)

        # Multiple iteration training
        total_loss = 0.0
        total_quality_loss = 0.0
        total_assoc_loss = 0.0

        self.model.train()

        for iter_idx in range(num_iterations):
            # Gradient clipping: prevent gradient explosion from BPTT being too long
            # Check if hidden state dimensions match
            h_in = None
            if sensor_id in self.hidden_states:
                B, M, K_plus_1, _ = hybrid_tensor.shape
                K = K_plus_1 - 1
                expected_size = B * M * K

                if self.hidden_states[sensor_id].shape[0] == expected_size:
                    h_in = self.hidden_states[sensor_id].detach()
                else:
                    h_in = None

            # Forward inference
            assoc_logits, quality_scores, h_out = self.model(hybrid_tensor, h_in)

            # Update internal GRU state for this sensor
            self.hidden_states[sensor_id] = h_out

            # Compute dual-teacher self-supervised loss
            loss, quality_loss, assoc_loss = self._compute_joint_loss(
                assoc_logits, quality_scores, measurements,
                predicted_measurements, predicted_variances
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
            self.optimizer.step()

            # Update EMA model
            if self.ema_model is not None:
                self._update_ema()

            total_loss += loss.item()
            total_quality_loss += quality_loss.item()
            total_assoc_loss += assoc_loss.item()

        # Average loss
        avg_loss = total_loss / num_iterations
        avg_quality_loss = total_quality_loss / num_iterations
        avg_assoc_loss = total_assoc_loss / num_iterations

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        # Record
        self.loss_history.append(avg_loss)
        self.quality_loss_history.append(avg_quality_loss)
        self.assoc_loss_history.append(avg_assoc_loss)

        # Format output for SLAM use
        with torch.no_grad():
            # Use EMA model for inference (if enabled)
            inference_model = self.ema_model if self.use_ema else self.model
            inference_model.eval()

            # Re-forward inference (using EMA model)
            eval_h_in = None
            if sensor_id in self.hidden_states:
                B, M, K_plus_1, _ = hybrid_tensor.shape
                K = K_plus_1 - 1
                expected_size = B * M * K
                if self.hidden_states[sensor_id].shape[0] == expected_size:
                    eval_h_in = self.hidden_states[sensor_id].detach()

            eval_assoc_logits, eval_quality_scores, _ = inference_model(hybrid_tensor, eval_h_in)

            # Temperature Scaling: T=0.1 for sharper output
            T = 0.1
            assoc_probs = F.softmax(eval_assoc_logits / T, dim=2).squeeze(0).cpu().numpy()  # (M, K)

            # Quality score: Sigmoid (M,)
            quality = eval_quality_scores.squeeze(0).cpu().numpy()

            # Convert dual-head output to SLAM format
            # Clutter probability (Dustbin) = 1.0 - Quality
            dustbin_probs = 1.0 - quality

            # No variance inflation - return ones (no scaling)
            final_scale = np.ones(len(dustbin_probs))

        return assoc_probs, dustbin_probs, final_scale, avg_loss

    def step_batch(self, sensor_data_list, num_iterations=1):
        """
        Execute one training step with batched data from multiple sensors

        This method properly handles multiple sensors by:
        1. Accumulating gradients from all sensors
        2. Updating weights only once per step
        3. Maintaining independent GRU states for each sensor
        4. Returning a single averaged loss

        Args:
            sensor_data_list: List of tuples, each containing:
                (sensor_id, hybrid_tensor, measurements, predicted_measurements, predicted_variances)
            num_iterations: Number of iterations per time step (default 1)

        Returns:
            results: List of tuples (sensor_id, assoc_probs, dustbin_probs, final_scale)
            avg_loss: Single averaged loss across all sensors
        """
        self.step_count += 1
        num_sensors = len(sensor_data_list)

        # Multiple iteration training
        total_loss = 0.0
        total_quality_loss = 0.0
        total_assoc_loss = 0.0

        self.model.train()

        for iter_idx in range(num_iterations):
            # Accumulate loss from all sensors
            accumulated_loss = 0.0
            accumulated_quality_loss = 0.0
            accumulated_assoc_loss = 0.0

            # Zero gradients once at the beginning
            self.optimizer.zero_grad()

            # Process each sensor
            for sensor_id, hybrid_tensor, measurements, predicted_measurements, predicted_variances in sensor_data_list:
                hybrid_tensor = hybrid_tensor.to(self.device)

                # Get sensor-specific hidden state
                h_in = None
                if sensor_id in self.hidden_states:
                    B, M, K_plus_1, _ = hybrid_tensor.shape
                    K = K_plus_1 - 1
                    expected_size = B * M * K

                    if self.hidden_states[sensor_id].shape[0] == expected_size:
                        h_in = self.hidden_states[sensor_id].detach()

                # Forward pass
                assoc_logits, quality_scores, h_out = self.model(hybrid_tensor, h_in)

                # Update sensor-specific GRU state
                self.hidden_states[sensor_id] = h_out

                # Compute loss for this sensor
                loss, quality_loss, assoc_loss = self._compute_joint_loss(
                    assoc_logits, quality_scores, measurements,
                    predicted_measurements, predicted_variances
                )

                # Accumulate losses (will be averaged later)
                accumulated_loss += loss
                accumulated_quality_loss += quality_loss.item()
                accumulated_assoc_loss += assoc_loss.item()

            # Average the accumulated loss across sensors
            avg_batch_loss = accumulated_loss / num_sensors

            # Backpropagation (only once for all sensors)
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.05)
            self.optimizer.step()

            # Update EMA model
            if self.ema_model is not None:
                self._update_ema()

            total_loss += avg_batch_loss.item()
            total_quality_loss += accumulated_quality_loss / num_sensors
            total_assoc_loss += accumulated_assoc_loss / num_sensors

        # Average loss across iterations
        avg_loss = total_loss / num_iterations
        avg_quality_loss = total_quality_loss / num_iterations
        avg_assoc_loss = total_assoc_loss / num_iterations

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        # Record
        self.loss_history.append(avg_loss)
        self.quality_loss_history.append(avg_quality_loss)
        self.assoc_loss_history.append(avg_assoc_loss)

        # Format output for SLAM use (inference for each sensor)
        results = []
        with torch.no_grad():
            inference_model = self.ema_model if self.use_ema else self.model
            inference_model.eval()

            for sensor_id, hybrid_tensor, measurements, predicted_measurements, predicted_variances in sensor_data_list:
                hybrid_tensor = hybrid_tensor.to(self.device)

                # Get sensor-specific hidden state for inference
                eval_h_in = None
                if sensor_id in self.hidden_states:
                    B, M, K_plus_1, _ = hybrid_tensor.shape
                    K = K_plus_1 - 1
                    expected_size = B * M * K
                    if self.hidden_states[sensor_id].shape[0] == expected_size:
                        eval_h_in = self.hidden_states[sensor_id].detach()

                # Inference
                eval_assoc_logits, eval_quality_scores, _ = inference_model(hybrid_tensor, eval_h_in)

                # Temperature Scaling: T=0.1 for sharper output
                T = 0.1
                assoc_probs = F.softmax(eval_assoc_logits / T, dim=2).squeeze(0).cpu().numpy()

                # Quality score
                quality = eval_quality_scores.squeeze(0).cpu().numpy()
                dustbin_probs = 1.0 - quality

                # No variance inflation
                final_scale = np.ones(len(dustbin_probs))

                results.append((sensor_id, assoc_probs, dustbin_probs, final_scale))

        return results, avg_loss

    def _compute_joint_loss(self, assoc_logits, quality_scores, measurements,
                           predicted_measurements, predicted_variances):
        """
        Compute joint loss: quality loss + association loss

        Dual-teacher self-supervised:
        1. Physics teacher: supervises quality head using RSS intrinsic consistency
        2. Geometry teacher: supervises association head using Hungarian algorithm
        """
        M = measurements.shape[1]
        K = predicted_measurements.shape[0]

        # Physics teacher: generate quality labels
        quality_loss = self._compute_quality_loss(quality_scores, measurements)

        # Geometry teacher: generate association labels (only train on high-quality measurements)
        assoc_loss = self._compute_association_loss(
            assoc_logits, quality_scores, measurements,
            predicted_measurements, predicted_variances
        )

        # Adaptive weight balancing
        if self.adaptive_weighting:
            quality_weight, assoc_weight = self._compute_adaptive_weights(
                quality_loss.item(), assoc_loss.item()
            )
        else:
            quality_weight = self.quality_weight
            assoc_weight = self.assoc_weight

        # Total loss
        total_loss = quality_weight * quality_loss + assoc_weight * assoc_loss

        # Record weight history
        self.weight_history.append({
            'quality_weight': quality_weight,
            'assoc_weight': assoc_weight
        })

        # Debug info
        if self.step_count % self.debug_interval == 0:
            print(f"\n[Loss Breakdown - Step {self.step_count}]")
            print(f"  Quality loss: {quality_loss.item():.4f} (weight={quality_weight:.3f})")
            print(f"  Association loss: {assoc_loss.item():.4f} (weight={assoc_weight:.3f})")
            print(f"  Total loss: {total_loss.item():.4f}")
            print(f"  Weighted contribution: quality={quality_weight * quality_loss.item():.4f}, assoc={assoc_weight * assoc_loss.item():.4f}")
            if self.scheduler is not None:
                print(f"  Current learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

        return total_loss, quality_loss, assoc_loss

    def _compute_adaptive_weights(self, quality_loss_val, assoc_loss_val):
        """
        Adaptive weight balancing

        Strategy:
        1. If a loss is too large, increase its weight
        2. Use exponential moving average to smooth weight changes
        3. Limit weight range to prevent extreme values
        """
        # Calculate loss ratio
        total = quality_loss_val + assoc_loss_val + 1e-8
        quality_ratio = quality_loss_val / total
        assoc_ratio = assoc_loss_val / total

        # Inverse proportional weight adjustment (larger loss gets larger weight)
        quality_weight = 1.0 + assoc_ratio
        assoc_weight = 1.0 + quality_ratio

        # Normalize (keep total weight constant)
        total_weight = quality_weight + assoc_weight
        quality_weight = quality_weight / total_weight * (self.quality_weight + self.assoc_weight)
        assoc_weight = assoc_weight / total_weight * (self.quality_weight + self.assoc_weight)

        # Limit weight range
        quality_weight = np.clip(quality_weight, 0.5, 3.0)
        assoc_weight = np.clip(assoc_weight, 0.5, 3.0)

        return quality_weight, assoc_weight

    def _compute_quality_loss(self, quality_scores, measurements):
        """
        Physics teacher: supervise quality head using RSS intrinsic consistency

        Three-interval fuzzy logic training strategy:
        - Interval 1: Core true region (RSS error<6dB) -> label=0.95, weight=1.0
        - Interval 2: Ambiguous region (6-15dB) -> label=0.5, weight=0.5
        - Interval 3: Core clutter region (>15dB) -> label=0.05, weight=1.0
        """
        M = measurements.shape[1]

        # Extract data
        z_dist = torch.from_numpy(measurements[0, :]).float().to(self.device)
        z_rss = torch.from_numpy(measurements[2, :]).float().to(self.device)

        # Physical model parameters
        P_tx = 15.41  # Transmit power (dBm)
        n = 2.0       # Path loss exponent

        # Calculate theoretical RSS
        rss_theory = P_tx - 10.0 * n * torch.log10(z_dist + 1e-6)

        # Calculate RSS error (dB)
        rss_error = torch.abs(z_rss - rss_theory)

        # Generate quality pseudo-labels
        target_quality = torch.zeros(M, device=self.device)
        quality_mask = torch.zeros(M, device=self.device)

        # Interval 1: Core true region (0-6dB)
        mask_core_true = (rss_error < 6.0)
        target_quality[mask_core_true] = 0.95
        quality_mask[mask_core_true] = 1.0

        # Interval 3: Core clutter region (>15dB)
        mask_core_clutter = (rss_error > 15.0)
        target_quality[mask_core_clutter] = 0.05
        quality_mask[mask_core_clutter] = 1.0

        # Interval 2: Ambiguous region (6-15dB)
        mask_ambiguous = (~mask_core_true) & (~mask_core_clutter)
        target_quality[mask_ambiguous] = 0.5
        quality_mask[mask_ambiguous] = 0.5

        # Debug info
        if self.step_count % self.debug_interval == 0:
            core_true_count = mask_core_true.sum().item()
            core_clutter_count = mask_core_clutter.sum().item()
            ambiguous_count = mask_ambiguous.sum().item()
            avg_quality = quality_scores.squeeze(0).mean().item()

            print(f"\n[Quality Head Diagnostics - Fuzzy Logic Three Intervals - Step {self.step_count}]")
            print(f"  Total measurements: {M}")
            print(f"  Interval 1: Core true (RSS error<6dB): {core_true_count} ({core_true_count/M*100:.1f}%)")
            print(f"  Interval 2: Ambiguous (6-15dB): {ambiguous_count} ({ambiguous_count/M*100:.1f}%)")
            print(f"  Interval 3: Core clutter (RSS error>15dB): {core_clutter_count} ({core_clutter_count/M*100:.1f}%)")
            print(f"  Average quality score: {avg_quality:.3f}")
            print(f"  RSS error range: [{rss_error.min().item():.1f}, {rss_error.max().item():.1f}] dB")

        # Calculate BCE loss
        if quality_mask.sum() > 0:
            loss = F.binary_cross_entropy(
                quality_scores.squeeze(0),
                target_quality,
                weight=quality_mask,
                reduction='sum'
            ) / quality_mask.sum()
        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss

    def _compute_association_loss(self, assoc_logits, quality_scores, measurements,
                                  predicted_measurements, predicted_variances):
        """
        Geometry teacher: supervise association head using Hungarian algorithm

        Key: only train association on high-quality measurements
        """
        M = measurements.shape[1]
        K = predicted_measurements.shape[0]

        # Extract high-quality measurements
        quality_np = quality_scores.squeeze(0).cpu().detach().numpy()
        valid_mask = quality_np > self.quality_threshold
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=self.device)

        # Extract subset of valid measurements
        z_dist = torch.from_numpy(measurements[0, valid_indices]).float().to(self.device)
        var_meas = torch.from_numpy(measurements[1, valid_indices]).float().to(self.device)

        z_pred = torch.from_numpy(predicted_measurements).float().to(self.device)
        var_pred = torch.from_numpy(predicted_variances).float().to(self.device)

        # Calculate geometric cost matrix
        joint_std = torch.sqrt(var_meas.unsqueeze(1) + var_pred.unsqueeze(0))
        diff_mat = torch.abs(z_dist.unsqueeze(1) - z_pred.unsqueeze(0))
        cost_matrix = diff_mat / (joint_std + 1e-6)

        # Hungarian algorithm generates matching labels
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # Gating: filter out matches with too large cost
        target_assoc = torch.full((len(valid_indices),), -1, dtype=torch.long, device=self.device)

        for i, r in enumerate(row_ind):
            if cost_np[r, col_ind[i]] < self.assoc_threshold:
                target_assoc[r] = col_ind[i]

        # Calculate association loss (only on successfully matched)
        match_mask = (target_assoc != -1)

        # Debug info
        if self.step_count % self.debug_interval == 0:
            matched_count = match_mask.sum().item()
            print(f"\n[Association Head Diagnostics - Step {self.step_count}]")
            print(f"  High-quality measurements: {len(valid_indices)}/{M} ({len(valid_indices)/M*100:.1f}%)")
            print(f"  Successfully matched (cost<{self.assoc_threshold}): {matched_count}/{len(valid_indices)} ({matched_count/max(len(valid_indices),1)*100:.1f}%)")
            if matched_count > 0:
                matched_costs = [cost_np[r, col_ind[i]] for i, r in enumerate(row_ind) if cost_np[r, col_ind[i]] < self.assoc_threshold]
                print(f"  Matching cost range: [{min(matched_costs):.2f}, {max(matched_costs):.2f}]")

        if match_mask.sum() > 0:
            sub_logits = assoc_logits.squeeze(0)[valid_indices]
            loss = F.cross_entropy(
                sub_logits[match_mask],
                target_assoc[match_mask],
                label_smoothing=0.1
            )
        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss

    def _update_ema(self):
        """Update EMA model"""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def save_checkpoint(self, checkpoint_path, epoch=None, additional_info=None):
        """Save model weights"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'loss_history': self.loss_history,
            'quality_loss_history': self.quality_loss_history,
            'assoc_loss_history': self.assoc_loss_history,
            'weight_history': self.weight_history,
            'quality_threshold': self.quality_threshold,
            'assoc_threshold': self.assoc_threshold,
            'aggregation': self.aggregation,
            'gamma': self.gamma,
            'edge_mode': self.edge_mode,
        }

        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if additional_info is not None:
            checkpoint['additional_info'] = additional_info

        torch.save(checkpoint, checkpoint_path)
        print(f"Dual-Head GNN V3 weights saved: {checkpoint_path}")
        print(f"  - Training steps: {self.step_count}")

    def load_checkpoint(self, checkpoint_path):
        """Load model weights"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"Warning: Weight file does not exist: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            if self.ema_model is not None and 'ema_model_state_dict' in checkpoint:
                self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.step_count = checkpoint.get('step_count', 0)
            self.loss_history = checkpoint.get('loss_history', [])
            self.quality_loss_history = checkpoint.get('quality_loss_history', [])
            self.assoc_loss_history = checkpoint.get('assoc_loss_history', [])
            self.weight_history = checkpoint.get('weight_history', [])

            print(f"Dual-Head GNN V3 weights loaded: {checkpoint_path}")
            print(f"  - Training steps: {self.step_count}")
            print(f"  - Loss history records: {len(self.loss_history)} data points")

            return True

        except Exception as e:
            print(f"Failed to load weights: {e}")
            return False

from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from rl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from rl_launcher.common.encoding import EncodingWrapper
from rl_launcher.common.optimizers import make_optimizer
from rl_launcher.common.typing import Batch, Data, Params, PRNGKey
from rl_launcher.networks.actor_critic_nets import Critic, Policy, GraspCritic, ensemblize
from rl_launcher.networks.lagrange import GeqLagrangeMultiplier
from rl_launcher.networks.mlp import MLP
from rl_launcher.utils.train_utils import _unpack


class SACAgentHybridDualArm(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
     
    Compared to SACAgent (in sac.py), this agent has a hybrid policy, with the gripper actions
    learned using DQN. Use this agent for dual arm setups.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )
    
    def forward_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="grasp_critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_grasp_critic(
            observations, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy( # type: ignore              
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for temperature Lagrange multiplier.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for Lagrange penalty for temperature.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]

        next_action_distributions = self.forward_policy(
            batch["next_observations"], rng=rng
        )
        
        next_actions, next_actions_log_probs = next_action_distributions.sample_and_log_prob(seed=rng)
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        # Extract continuous actions for critic
        actions = jnp.concatenate([batch["actions"][..., :6], batch["actions"][..., 7:13]], axis=-1)

        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info
    

    def grasp_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""

        batch_size = batch["rewards"].shape[0]
        grasp_action1 = jnp.round(batch["actions"][..., 6]).astype(jnp.int16) + 1  # Cast env action from [-1, 1] to {0, 1, 2}
        grasp_action2 = jnp.round(batch["actions"][..., 13]).astype(jnp.int16) + 1  # Cast env action from [-1, 1] to {0, 1, 2}
        
        # Combine the two grasp actions into a single joint action ranging from 0 to 8
        joint_grasp_action = grasp_action1 * 3 + grasp_action2  # 0 ≤joint_grasp_action < 9
        # Ensure joint actions are within the valid range
        chex.assert_shape(joint_grasp_action, (batch_size,))
        
        # Evaluate next grasp Qs for all ensemble members (forward pass)
        target_next_grasp_qs = self.forward_target_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        chex.assert_shape(target_next_grasp_qs, (batch_size, 9))
        
        # Select target next grasp Q based on the joint action that maximizes the current grasp Q
        next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        # For DQN, select actions using online network, evaluate with target network
        best_next_grasp_action = next_grasp_qs.argmax(axis=-1) 
        chex.assert_shape(best_next_grasp_action, (batch_size,))
        
        target_next_grasp_q = target_next_grasp_qs[jnp.arange(batch_size), best_next_grasp_action]
        chex.assert_shape(target_next_grasp_q, (batch_size,))
        
        # Compute target Q-values
        grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
        target_grasp_q = (
            grasp_rewards
            + self.config["discount"] * batch["masks"] * target_next_grasp_q
        )
        chex.assert_shape(target_grasp_q, (batch_size,))
        
        # Forward pass through the online grasp critic to get predicted Q-values for all joint actions
        predicted_grasp_qs = self.forward_grasp_critic(
            batch["observations"], 
            rng=rng, 
            grad_params=params
        )
        chex.assert_shape(predicted_grasp_qs, (batch_size, 9))
        
        # Select the predicted Q-values for the taken joint grasp actions in the batch
        predicted_grasp_q = predicted_grasp_qs[jnp.arange(batch_size), joint_grasp_action]
        chex.assert_shape(predicted_grasp_q, (batch_size,))
        
        # Compute MSE loss between predicted and target Q-values
        chex.assert_equal_shape([predicted_grasp_q, target_grasp_q])
        grasp_critic_loss = jnp.mean((predicted_grasp_q - target_grasp_q) ** 2)
        
        info = {
            "grasp_critic_loss": grasp_critic_loss,
            "predicted_grasp_qs": jnp.mean(predicted_grasp_q),
            "target_grasp_qs": jnp.mean(target_grasp_q),
            "grasp_rewards": grasp_rewards.mean(),
        }

        return grasp_critic_loss, info


    def bc_policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Behavioral cloning loss for offline pretraining of the actor.
        Regresses the 12-dim continuous action (gripper handled separately by grasp_critic).
        """
        batch_size = batch["rewards"].shape[0]
        actor_loss_type = self.config.get("bc_actor_loss_type", "mse")
        huber_delta = self.config.get("bc_huber_delta", 0.10)
        active_xyz_threshold = self.config.get("bc_active_xyz_threshold", 0.05)
        active_xyz_weight = self.config.get("bc_active_xyz_weight", 4.0)
        high_xyz_norm_beta = self.config.get("bc_high_xyz_norm_beta", 0.0)
        rot_mse_weight = self.config.get("bc_rot_mse_weight", 0.10)
        xyz_norm_loss_weight = self.config.get("bc_xyz_norm_loss_weight", 0.50)
        xyz_relative_norm_loss_weight = self.config.get(
            "bc_xyz_relative_norm_loss_weight", 0.15
        )
        rot_norm_loss_weight = self.config.get("bc_rot_norm_loss_weight", 0.02)
        xyz_cosine_loss_weight = self.config.get("bc_xyz_cosine_loss_weight", 0.02)
        rot_cosine_loss_weight = self.config.get("bc_rot_cosine_loss_weight", 0.0)
        inactive_xyz_loss_weight = self.config.get("bc_inactive_xyz_loss_weight", 0.10)
        eps = 1e-6

        # Extract continuous 12-dim from demo actions (skip gripper at indices 6 and 13)
        continuous_actions = jnp.concatenate([
            batch["actions"][..., :6],
            batch["actions"][..., 7:13]
        ], axis=-1)
        chex.assert_shape(continuous_actions, (batch_size, 12))

        # Forward actor
        action_distributions = self.forward_policy(
            batch["observations"], rng=rng, grad_params=params
        )

        # Clip target to open interval for tanh-squash distribution
        target_actions = jnp.clip(continuous_actions, -1.0 + 1e-6, 1.0 - 1e-6)

        # Track the raw action-space mode directly. Physical scaling is applied
        # only by the env at execution time, not in the BC target/loss space.
        predicted_actions = action_distributions.mode()

        pred_left_xyz = predicted_actions[..., :3]
        pred_left_rot = predicted_actions[..., 3:6]
        pred_right_xyz = predicted_actions[..., 6:9]
        pred_right_rot = predicted_actions[..., 9:12]
        target_left_xyz = target_actions[..., :3]
        target_left_rot = target_actions[..., 3:6]
        target_right_xyz = target_actions[..., 6:9]
        target_right_rot = target_actions[..., 9:12]
        pred_xyz = jnp.concatenate([pred_left_xyz, pred_right_xyz], axis=-1)
        target_xyz = jnp.concatenate([target_left_xyz, target_right_xyz], axis=-1)
        pred_rot = jnp.concatenate([pred_left_rot, pred_right_rot], axis=-1)
        target_rot = jnp.concatenate([target_left_rot, target_right_rot], axis=-1)

        target_norm = jnp.linalg.norm(target_actions, axis=-1)
        pred_norm = jnp.linalg.norm(predicted_actions, axis=-1)
        target_left_xyz_norm = jnp.linalg.norm(target_left_xyz, axis=-1)
        target_right_xyz_norm = jnp.linalg.norm(target_right_xyz, axis=-1)
        pred_left_xyz_norm = jnp.linalg.norm(pred_left_xyz, axis=-1)
        pred_right_xyz_norm = jnp.linalg.norm(pred_right_xyz, axis=-1)
        target_xyz_norm = jnp.linalg.norm(target_xyz, axis=-1)
        pred_xyz_norm = jnp.linalg.norm(pred_xyz, axis=-1)
        target_left_rot_norm = jnp.linalg.norm(target_left_rot, axis=-1)
        target_right_rot_norm = jnp.linalg.norm(target_right_rot, axis=-1)
        pred_left_rot_norm = jnp.linalg.norm(pred_left_rot, axis=-1)
        pred_right_rot_norm = jnp.linalg.norm(pred_right_rot, axis=-1)
        target_rot_norm = jnp.linalg.norm(target_rot, axis=-1)
        pred_rot_norm = jnp.linalg.norm(pred_rot, axis=-1)

        left_active_mask = (target_left_xyz_norm > active_xyz_threshold).astype(jnp.float32)
        right_active_mask = (target_right_xyz_norm > active_xyz_threshold).astype(jnp.float32)
        left_weights = (
            1.0
            + active_xyz_weight * left_active_mask
            + high_xyz_norm_beta * target_left_xyz_norm
        )
        right_weights = (
            1.0
            + active_xyz_weight * right_active_mask
            + high_xyz_norm_beta * target_right_xyz_norm
        )
        left_weight_sum = jnp.sum(left_weights) + eps
        right_weight_sum = jnp.sum(right_weights) + eps

        if actor_loss_type == "huber":
            per_dim_left_xyz_loss = optax.huber_loss(
                pred_left_xyz - target_left_xyz, delta=huber_delta
            )
            per_dim_right_xyz_loss = optax.huber_loss(
                pred_right_xyz - target_right_xyz, delta=huber_delta
            )
            per_dim_left_rot_loss = optax.huber_loss(
                pred_left_rot - target_left_rot, delta=huber_delta
            )
            per_dim_right_rot_loss = optax.huber_loss(
                pred_right_rot - target_right_rot, delta=huber_delta
            )
        else:
            per_dim_left_xyz_loss = (pred_left_xyz - target_left_xyz) ** 2
            per_dim_right_xyz_loss = (pred_right_xyz - target_right_xyz) ** 2
            per_dim_left_rot_loss = (pred_left_rot - target_left_rot) ** 2
            per_dim_right_rot_loss = (pred_right_rot - target_right_rot) ** 2

        left_xyz_regression_loss = (
            jnp.sum(left_weights * jnp.mean(per_dim_left_xyz_loss, axis=-1))
            / left_weight_sum
        )
        right_xyz_regression_loss = (
            jnp.sum(right_weights * jnp.mean(per_dim_right_xyz_loss, axis=-1))
            / right_weight_sum
        )
        xyz_regression_loss = 0.5 * (
            left_xyz_regression_loss + right_xyz_regression_loss
        )
        left_rot_regression_loss = (
            jnp.sum(left_weights * jnp.mean(per_dim_left_rot_loss, axis=-1))
            / left_weight_sum
        )
        right_rot_regression_loss = (
            jnp.sum(right_weights * jnp.mean(per_dim_right_rot_loss, axis=-1))
            / right_weight_sum
        )
        rot_regression_loss = 0.5 * (
            left_rot_regression_loss + right_rot_regression_loss
        )
        action_regression_loss = xyz_regression_loss + rot_mse_weight * rot_regression_loss
        left_xyz_mse_loss = (
            jnp.sum(left_weights * jnp.mean((pred_left_xyz - target_left_xyz) ** 2, axis=-1))
            / left_weight_sum
        )
        right_xyz_mse_loss = (
            jnp.sum(right_weights * jnp.mean((pred_right_xyz - target_right_xyz) ** 2, axis=-1))
            / right_weight_sum
        )
        xyz_mse_loss = 0.5 * (left_xyz_mse_loss + right_xyz_mse_loss)
        left_rot_mse_loss = (
            jnp.sum(left_weights * jnp.mean((pred_left_rot - target_left_rot) ** 2, axis=-1))
            / left_weight_sum
        )
        right_rot_mse_loss = (
            jnp.sum(right_weights * jnp.mean((pred_right_rot - target_right_rot) ** 2, axis=-1))
            / right_weight_sum
        )
        rot_mse_loss = 0.5 * (left_rot_mse_loss + right_rot_mse_loss)

        left_xyz_norm_loss = (
            jnp.sum(left_weights * (pred_left_xyz_norm - target_left_xyz_norm) ** 2)
            / left_weight_sum
        )
        right_xyz_norm_loss = (
            jnp.sum(right_weights * (pred_right_xyz_norm - target_right_xyz_norm) ** 2)
            / right_weight_sum
        )
        xyz_norm_loss = 0.5 * (left_xyz_norm_loss + right_xyz_norm_loss)
        left_rot_norm_loss = (
            jnp.sum(left_weights * (pred_left_rot_norm - target_left_rot_norm) ** 2)
            / left_weight_sum
        )
        right_rot_norm_loss = (
            jnp.sum(right_weights * (pred_right_rot_norm - target_right_rot_norm) ** 2)
            / right_weight_sum
        )
        rot_norm_loss = 0.5 * (left_rot_norm_loss + right_rot_norm_loss)

        left_xyz_relative_norm_loss = jnp.sum(
            left_weights
            * left_active_mask
            * (
                (pred_left_xyz_norm - target_left_xyz_norm)
                / jnp.maximum(target_left_xyz_norm, active_xyz_threshold)
            ) ** 2
        ) / (jnp.sum(left_weights * left_active_mask) + eps)
        right_xyz_relative_norm_loss = jnp.sum(
            right_weights
            * right_active_mask
            * (
                (pred_right_xyz_norm - target_right_xyz_norm)
                / jnp.maximum(target_right_xyz_norm, active_xyz_threshold)
            ) ** 2
        ) / (jnp.sum(right_weights * right_active_mask) + eps)
        xyz_relative_norm_loss = 0.5 * (
            left_xyz_relative_norm_loss + right_xyz_relative_norm_loss
        )
        active_xyz_count = jnp.sum(left_active_mask) + jnp.sum(right_active_mask) + eps
        xyz_norm_ratio = (
            jnp.sum(left_active_mask * pred_left_xyz_norm / jnp.maximum(target_left_xyz_norm, eps))
            + jnp.sum(right_active_mask * pred_right_xyz_norm / jnp.maximum(target_right_xyz_norm, eps))
        ) / active_xyz_count

        left_inactive_mask = 1.0 - left_active_mask
        right_inactive_mask = 1.0 - right_active_mask
        left_inactive_xyz_loss = jnp.sum(
            left_inactive_mask * (pred_left_xyz_norm ** 2)
        ) / (jnp.sum(left_inactive_mask) + eps)
        right_inactive_xyz_loss = jnp.sum(
            right_inactive_mask * (pred_right_xyz_norm ** 2)
        ) / (jnp.sum(right_inactive_mask) + eps)
        inactive_xyz_loss = 0.5 * (
            left_inactive_xyz_loss + right_inactive_xyz_loss
        )

        left_xyz_cosine = jnp.sum(pred_left_xyz * target_left_xyz, axis=-1) / (
            pred_left_xyz_norm * target_left_xyz_norm + eps
        )
        right_xyz_cosine = jnp.sum(pred_right_xyz * target_right_xyz, axis=-1) / (
            pred_right_xyz_norm * target_right_xyz_norm + eps
        )
        left_xyz_cosine = jnp.clip(left_xyz_cosine, -1.0, 1.0)
        right_xyz_cosine = jnp.clip(right_xyz_cosine, -1.0, 1.0)
        left_xyz_cosine_mask = (target_left_xyz_norm > 1e-3).astype(jnp.float32)
        right_xyz_cosine_mask = (target_right_xyz_norm > 1e-3).astype(jnp.float32)
        left_xyz_cosine_loss = (
            jnp.sum(left_weights * left_xyz_cosine_mask * (1.0 - left_xyz_cosine))
            / (jnp.sum(left_weights * left_xyz_cosine_mask) + eps)
        )
        right_xyz_cosine_loss = (
            jnp.sum(right_weights * right_xyz_cosine_mask * (1.0 - right_xyz_cosine))
            / (jnp.sum(right_weights * right_xyz_cosine_mask) + eps)
        )
        xyz_cosine_loss = 0.5 * (left_xyz_cosine_loss + right_xyz_cosine_loss)

        rot_cosine = jnp.sum(pred_rot * target_rot, axis=-1) / (
            pred_rot_norm * target_rot_norm + eps
        )
        rot_cosine = jnp.clip(rot_cosine, -1.0, 1.0)
        rot_cosine_mask = (target_rot_norm > 1e-3).astype(jnp.float32)
        # Use average of left and right weights for combined rotation cosine loss
        sample_weights = 0.5 * (left_weights + right_weights)
        rot_cosine_weight_sum = jnp.sum(sample_weights * rot_cosine_mask) + eps
        rot_cosine_loss = (
            jnp.sum(sample_weights * rot_cosine_mask * (1.0 - rot_cosine))
            / rot_cosine_weight_sum
        )

        bc_loss = (
            action_regression_loss
            + xyz_norm_loss_weight * xyz_norm_loss
            + xyz_relative_norm_loss_weight * xyz_relative_norm_loss
            + rot_norm_loss_weight * rot_norm_loss
            + xyz_cosine_loss_weight * xyz_cosine_loss
            + rot_cosine_loss_weight * rot_cosine_loss
            + inactive_xyz_loss_weight * inactive_xyz_loss
        )

        # Metrics for monitoring.
        log_probs = action_distributions.log_prob(target_actions)
        mse = ((predicted_actions - target_actions) ** 2).sum(-1).mean()
        mse_per_dim = ((predicted_actions - target_actions) ** 2).mean()
        xyz_demo_mse = ((pred_xyz - target_xyz) ** 2).mean()
        rot_mse = ((pred_rot - target_rot) ** 2).mean()

        info = {
            "actor_loss": bc_loss,
            "bc_loss": bc_loss,
            "bc_action_regression_loss": action_regression_loss,
            "bc_xyz_regression_loss": xyz_regression_loss,
            "bc_left_xyz_regression_loss": left_xyz_regression_loss,
            "bc_right_xyz_regression_loss": right_xyz_regression_loss,
            "bc_rot_regression_loss": rot_regression_loss,
            "bc_action_mse_loss": xyz_mse_loss + rot_mse_weight * rot_mse_loss,
            "bc_xyz_mse_loss": xyz_mse_loss,
            "bc_left_xyz_mse_loss": left_xyz_mse_loss,
            "bc_right_xyz_mse_loss": right_xyz_mse_loss,
            "bc_rot_mse_loss": rot_mse_loss,
            "bc_norm_loss": xyz_norm_loss,
            "bc_xyz_norm_loss": xyz_norm_loss,
            "bc_xyz_relative_norm_loss": xyz_relative_norm_loss,
            "bc_inactive_xyz_loss": inactive_xyz_loss,
            "bc_left_inactive_xyz_loss": left_inactive_xyz_loss,
            "bc_right_inactive_xyz_loss": right_inactive_xyz_loss,
            "bc_rot_norm_loss": rot_norm_loss,
            "bc_cosine_loss": xyz_cosine_loss,
            "bc_xyz_cosine_loss": xyz_cosine_loss,
            "bc_rot_cosine_loss": rot_cosine_loss,
            "bc_nll": -log_probs.mean(),
            "bc_mse": mse,
            "bc_mse_per_dim": mse_per_dim,
            "bc_xyz_demo_mse": xyz_demo_mse,
            "bc_rot_mse": rot_mse,
            "bc_action_norm_pred": pred_norm.mean(),
            "bc_action_norm_target": target_norm.mean(),
            "bc_xyz_action_norm_pred": pred_xyz_norm.mean(),
            "bc_xyz_action_norm_target": target_xyz_norm.mean(),
            "bc_rot_action_norm_pred": pred_rot_norm.mean(),
            "bc_rot_action_norm_target": target_rot_norm.mean(),
            "bc_xyz_norm_ratio": xyz_norm_ratio,
            "bc_active_xyz_ratio": 0.5 * (
                left_active_mask.mean() + right_active_mask.mean()
            ),
            "bc_left_active_xyz_ratio": left_active_mask.mean(),
            "bc_right_active_xyz_ratio": right_active_mask.mean(),
            "bc_sample_weight_mean": 0.5 * (
                left_weights.mean() + right_weights.mean()
            ),
        }

        return bc_loss, info

    def bc_grasp_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """Behavioral cloning loss for the dual-gripper joint action."""
        batch_size = batch["rewards"].shape[0]
        grasp_action1 = jnp.round(batch["actions"][..., 6]).astype(jnp.int32) + 1
        grasp_action2 = jnp.round(batch["actions"][..., 13]).astype(jnp.int32) + 1
        grasp_action1 = jnp.clip(grasp_action1, 0, 2)
        grasp_action2 = jnp.clip(grasp_action2, 0, 2)
        joint_grasp_action = grasp_action1 * 3 + grasp_action2
        chex.assert_shape(joint_grasp_action, (batch_size,))

        logits = self.forward_grasp_critic(
            batch["observations"],
            rng=rng,
            grad_params=params,
        )
        chex.assert_shape(logits, (batch_size, 9))

        one_hot = jax.nn.one_hot(joint_grasp_action, 9)
        grasp_bc_loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        pred_joint = logits.argmax(axis=-1)
        pred_grasp1 = pred_joint // 3
        pred_grasp2 = pred_joint % 3

        info = {
            "grasp_critic_loss": grasp_bc_loss,
            "bc_grasp_loss": grasp_bc_loss,
            "bc_grasp_joint_accuracy": jnp.mean(pred_joint == joint_grasp_action),
            "bc_grasp_left_accuracy": jnp.mean(pred_grasp1 == grasp_action1),
            "bc_grasp_right_accuracy": jnp.mean(pred_grasp2 == grasp_action2),
        }

        return grasp_bc_loss, info

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}
    

    def loss_fns(self, batch, bc_mode=False):
        """
        Return loss functions for each network.

        Args:
            batch: Training batch
            bc_mode: If True, use BC loss for actor instead of SAC policy loss
        """
        actor_loss_fn = partial(self.bc_policy_loss_fn, batch) if bc_mode else partial(self.policy_loss_fn, batch)
        grasp_loss_fn = (
            partial(self.bc_grasp_loss_fn, batch)
            if bc_mode
            else partial(self.grasp_critic_loss_fn, batch)
        )

        return {
            "critic": partial(self.critic_loss_fn, batch),
            "grasp_critic": grasp_loss_fn,
            "actor": actor_loss_fn,
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update", "bc_mode"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "grasp_critic", "temperature"}
        ),
        bc_mode: bool = False,
        **kwargs
    ) -> Tuple["SACAgentHybridDualArm", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
            bc_mode: If True, train actor with BC loss instead of SAC policy loss.
        Returns:
            Tuple of (new agent, info dict).
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 14))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch, bc_mode=bc_mode, **kwargs)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update or "grasp_critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """

        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            ee_actions = dist.mode()
        else:
            ee_actions = dist.sample(seed=seed)
        
        seed, grasp_key = jax.random.split(seed, 2)
        grasp_q_values = self.forward_grasp_critic(observations, rng=grasp_key, train=False)  # (batch_size, 9)
        
        # Select grasp actions based on the joint grasp Q-values
        joint_grasp_action = grasp_q_values.argmax(axis=-1)

        # Decompose joint action back into two individual actions
        grasp_action1 = joint_grasp_action // 3 - 1  # Mapping back to {-1, 0, 1}
        grasp_action2 = joint_grasp_action % 3 - 1  # Mapping back to {-1, 0, 1}

        # Combine continuous actions with grasp actions
        return jnp.concatenate([
            ee_actions[..., :6],
            grasp_action1[..., None],
            ee_actions[..., 6:],
            grasp_action2[..., None]
        ], axis=-1)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        grasp_critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        grasp_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "grasp_critic": grasp_critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "grasp_critic": make_optimizer(**grasp_critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)

        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions[..., jnp.concatenate([jnp.arange(6), jnp.arange(7, 13)])]],
            grasp_critic=[observations],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs: dict = {
            "hidden_dims": [128, 128],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "resnet":
            from rl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from rl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
            "grasp_critic": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")
        
        grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
        grasp_critic_def = partial(
            GraspCritic, encoder=encoders["grasp_critic"], network=grasp_critic_backbone, output_dim=9
        )(name="grasp_critic")
        
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=12,  # 6 continuous actions for each arm
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            grasp_critic_def=grasp_critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from rl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent

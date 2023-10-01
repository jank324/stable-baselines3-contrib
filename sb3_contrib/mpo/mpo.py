from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn import functional as F

from sb3_contrib.mpo.policies import Actor, CnnPolicy, MlpPolicy, MPOPolicy, MultiInputPolicy

SelfMPO = TypeVar("SelfMPO", bound="MPO")

FLOAT_EPSILON = 1e-8


class MPO(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3) TODO: Update docstring
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: MPOPolicy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(  # TODO: Copy params from Acme
        self,
        policy: Union[str, Type[MPOPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (
            1,
            "episode",
        ),  # TODO: Tonic step_between_batches=50
        gradient_steps: int = -1,  # TODO: Tonic batch_iterations=50
        dual_optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        dual_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        per_dim_constraining: bool = True,
        initial_log_temperature: float = 1.0,
        initial_log_alpha_mean: float = 1.0,
        initial_log_alpha_std: float = 10.0,
        action_penalization: bool = True,
        num_samples: int = 20,
        epsilon: float = 0.1,
        epsilon_penalty: float = 1e-3,
        epsilon_mean: float = 1e-3,
        epsilon_std: float = 1e-6,
        gradient_clip: float = 0.0,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.dual_optimizer_class = dual_optimizer_class
        self.dual_optimizer_kwargs = (
            dual_optimizer_kwargs if dual_optimizer_kwargs is not None else {"lr": 1e-2}
        )  # Adam with lr=1e-2 is default in Acme
        self.per_dim_constraining = per_dim_constraining
        self.initial_log_temperature = initial_log_temperature
        self.initial_log_alpha_mean = initial_log_alpha_mean
        self.initial_log_alpha_std = initial_log_alpha_std
        self.action_penalization = action_penalization
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.epsilon_penalty = epsilon_penalty
        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std
        self.gradient_clip = gradient_clip

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        self._create_aliases()
        self._setup_dual_variables_and_optimizer()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _setup_dual_variables_and_optimizer(self) -> None:
        """
        Setup dual variables and optimiser.
        """
        shape = [self.action_space.shape[0]] if self.per_dim_constraining else [1]  # type: ignore

        self.log_temperature = nn.Parameter(th.as_tensor(self.initial_log_temperature, dtype=th.float32))
        self.log_alpha_mean = nn.Parameter(th.full(shape, self.initial_log_alpha_mean, dtype=th.float32))
        self.log_alpha_std = nn.Parameter(th.full(shape, self.initial_log_alpha_std, dtype=th.float32))

        self.dual_variables = [
            self.log_temperature,
            self.log_alpha_mean,
            self.log_alpha_std,
        ]

        if self.action_penalization:
            self.log_penalty_temperature = nn.Parameter(th.as_tensor(self.initial_log_temperature, dtype=th.float32))
            self.dual_variables.append(self.log_penalty_temperature)

        self.dual_optimizer = self.dual_optimizer_class(self.dual_variables, **self.dual_optimizer_kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.dual_optimizer])

        critic_losses = []
        policy_losses, kl_losses, dual_losses, actor_losses = [], [], [], []
        logged_temperatures, logged_alpha_means, logged_alpha_stds = [], [], []
        if self.action_penalization:
            logged_penalties = []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                target_distributions = self.actor_target.predict_action_distribution(
                    replay_data.next_observations
                ).distribution
                next_action_samples = target_distributions.sample((self.num_samples,))

                tiled_observations = replay_data.observations.tile(
                    self.num_samples,
                    *(1 for _ in range(replay_data.observations.dim())),
                )
                tiled_next_observations = replay_data.next_observations.tile(
                    self.num_samples,
                    *(1 for _ in range(replay_data.next_observations.dim())),
                )

                # Flatten sample and batch dimensions for critic evaluations
                flat_observations = merge_first_two_dims(tiled_observations)
                flat_next_observations = merge_first_two_dims(tiled_next_observations)
                flat_actions = merge_first_two_dims(next_action_samples)

                # Use q1_forward because MPO defaults to using only one critic
                flat_values = self.critic_target.q1_forward(flat_observations, flat_actions)
                flat_next_values = self.critic_target.q1_forward(flat_next_observations, flat_actions)

                # Restore sample and batch dimensions
                values = flat_values.view(self.num_samples, -1, 1)
                next_values = flat_next_values.view(self.num_samples, -1, 1)

                target_distributions = Independent(target_distributions, -1)

                returns = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_values.mean(dim=0)

            self.actor.optimizer.zero_grad()
            self.dual_optimizer.zero_grad()
            self.critic.optimizer.zero_grad()

            distributions = self.actor.predict_action_distribution(replay_data.observations).distribution
            distributions = Independent(distributions, -1)

            temperature = F.softplus(self.log_temperature) + FLOAT_EPSILON
            alpha_mean = F.softplus(self.log_alpha_mean) + FLOAT_EPSILON
            alpha_std = F.softplus(self.log_alpha_std) + FLOAT_EPSILON
            weights, temperature_loss = weights_and_temperature_loss(values, self.epsilon, temperature)

            # Action penalization is quadratic beyond [-1, 1].
            if self.action_penalization:
                penalty_temperature = F.softplus(self.log_penalty_temperature) + FLOAT_EPSILON
                diff_bounds = next_action_samples - th.clamp(next_action_samples, -1, 1)
                action_bound_costs = -th.norm(diff_bounds, dim=-1).unsqueeze(-1)
                penalty_weights, penalty_temperature_loss = weights_and_temperature_loss(
                    action_bound_costs, self.epsilon_penalty, penalty_temperature
                )
                weights += penalty_weights
                temperature_loss += penalty_temperature_loss

            # Decompose the policy into fixed-mean and fixed-std distributions
            fixed_std_distribution = Independent(Normal(distributions.mean, target_distributions.stddev), -1)
            fixed_mean_distribution = Independent(Normal(target_distributions.mean, distributions.stddev), -1)

            # Compute the decomposed policy losses
            policy_mean_losses = (
                fixed_std_distribution.base_dist.log_prob(next_action_samples).sum(dim=-1).unsqueeze(-1) * weights
            ).sum(dim=0)
            policy_mean_loss = -(policy_mean_losses).mean()
            policy_std_losses = (
                fixed_mean_distribution.base_dist.log_prob(next_action_samples).sum(dim=-1).unsqueeze(-1) * weights
            ).sum(dim=0)
            policy_std_loss = -policy_std_losses.mean()

            # Compute the decomposed KL between the target and online policies
            if self.per_dim_constraining:
                kl_mean = th.distributions.kl.kl_divergence(target_distributions.base_dist, fixed_std_distribution.base_dist)
                kl_std = th.distributions.kl.kl_divergence(target_distributions.base_dist, fixed_mean_distribution.base_dist)
            else:
                kl_mean = th.distributions.kl.kl_divergence(target_distributions, fixed_std_distribution)
                kl_std = th.distributions.kl.kl_divergence(target_distributions, fixed_mean_distribution)

            # Compute the alpha-weighted KL-penalty and dual losses
            kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(kl_mean, alpha_mean, self.epsilon_mean)
            kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(kl_std, alpha_std, self.epsilon_std)

            # Combine losses
            policy_loss = policy_mean_loss + policy_std_loss
            kl_loss = kl_mean_loss + kl_std_loss
            dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
            actor_loss = policy_loss + kl_loss + dual_loss

            # Compute the critic loss
            critic_values = self.critic.q1_forward(replay_data.observations, replay_data.actions)
            critic_loss = F.mse_loss(critic_values, returns)

            actor_loss.backward()
            critic_loss.backward()
            if self.gradient_clip > 0:
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
                th.nn.utils.clip_grad_norm_(self.dual_variables, self.gradient_clip)
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
            self.actor.optimizer.step()
            self.dual_optimizer.step()
            self.critic.optimizer.step()

            # Update the target networks
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

            # Save losses for logging
            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())
            policy_losses.append(policy_loss.item())
            kl_losses.append(kl_loss.item())
            dual_losses.append(dual_loss.item())

            # Save dual variables for logging
            logged_temperatures.append(temperature.mean().item())
            logged_alpha_means.append(alpha_mean.mean().item())
            logged_alpha_stds.append(alpha_std.mean().item())
            if self.action_penalization:
                logged_penalties.append(penalty_temperature.mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/dual_loss", np.mean(dual_losses))
        self.logger.record("train/temperature", np.mean(logged_temperatures))
        self.logger.record("train/alpha_mean", np.mean(logged_alpha_means))
        self.logger.record("train/alpha_std", np.mean(logged_alpha_stds))
        if self.action_penalization:
            self.logger.record("train/penalty_temperature", np.mean(logged_penalties))
        if hasattr(self.actor, "log_std"):
            self.logger.record("train/std", th.exp(self.actor.log_std).mean().item())

    def learn(
        self: SelfMPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "MPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfMPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [  # noqa: RUF005
            "actor",
            "critic",
            "actor_target",
            "critic_target",
            "dual_variables",
            "log_temperature",
            "log_alpha_mean",
            "log_alpha_std",
            "log_penalty_temperature",
            "dual_optimizer",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "dual_optimizer"]
        other = ["log_temperature", "log_alpha_mean", "log_alpha_std", "log_penalty_temperature"]
        return state_dicts, other


def merge_first_two_dims(original: th.Tensor) -> th.Tensor:
    """
    Merges the first two dimensions of a tensor, e.g. a tensor of shape (a, b, c, d) will be reshaped to (a*b, c, d).
    """
    merged_dim_size = original.shape[0] * original.shape[1]
    merged_tensor = original.view(merged_dim_size, *original.shape[2:])

    return merged_tensor


def weights_and_temperature_loss(q_values: th.Tensor, epsilon: float, temperature: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    Computes weights for loss computation and the temperature loss.

    Following implementation in: https://github.com/fabiopardo/tonic/blob/master/tonic/torch/updaters/actors.py
    """
    tempered_q_values = q_values.detach() / temperature
    weights = F.softmax(tempered_q_values, dim=0)
    weights = weights.detach()

    # Temperature loss (dual of the E-step).
    q_log_sum_exp = th.logsumexp(tempered_q_values, dim=0)
    num_actions = th.as_tensor(q_values.shape[0], dtype=th.float32)
    log_num_actions = th.log(num_actions)
    loss = epsilon + (q_log_sum_exp).mean() - log_num_actions
    loss = temperature * loss

    return weights, loss


def parametric_kl_and_dual_losses(kl: th.Tensor, alpha: th.Tensor, epsilon: float) -> Tuple[th.Tensor, th.Tensor]:
    """
    Computes the alpha-weighted KL-penalty and dual losses.
    """
    kl_mean = kl.mean(dim=0)
    kl_loss = (alpha.detach() * kl_mean).sum()
    alpha_loss = (alpha * (epsilon - kl_mean.detach())).sum()
    return kl_loss, alpha_loss

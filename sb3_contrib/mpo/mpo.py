from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, ReplayBufferSamples, Schedule
from stable_baselines3.common.utils import get_parameters_by_name
from torch import nn
from torch.distributions import Independent, Normal
from torch.nn import functional as F

from sb3_contrib.mpo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

SelfMPO = TypeVar("SelfMPO", bound="MPO")

FLOAT_EPSILON = 1e-8


class MPO(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
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

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        dual_optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        dual_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
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
            action_noise=action_noise,
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

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.dual_optimizer_class = dual_optimizer_class
        self.dual_optimizer_kwargs = dual_optimizer_kwargs or {"lr": 1e-2}  # Adam with lr=1e-2 is default in Acme

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # TODO: This setup as copied from TD3 is a little odd
        self._setup_dual_variables_and_optimizer()

    def _setup_dual_variables_and_optimizer(self) -> None:
        """
        Setup dual variables and optimiser.
        """
        shape = [self.action_space.shape[0]] if self.per_dim_constraining else [1]

        self.log_temperature = nn.Parameter(th.as_tensor([self.initial_log_temperature], dtype=th.float32))
        self.log_alpha_mean = nn.Parameter(th.full(shape, self.initial_log_alpha_mean, dtype=th.float32))
        self.log_alpha_std = nn.Parameter(th.full(shape, self.initial_log_alpha_std, dtype=th.float32))

        self.dual_variables = [self.log_temperature, self.log_alpha_mean, self.log_alpha_std]

        if self.action_penalization:
            self.log_penalty_temperature = nn.Parameter(th.as_tensor([self.initial_log_temperature], dtype=th.float32))
            self.dual_variables.append(self.log_penalty_temperature)

        self.dual_optimizer = self.dual_optimizer_class(self.dual_variables, **self.dual_optimizer_kwargs)

    def _train_critic(self, replay_data: ReplayBufferSamples) -> Dict[str, th.Tensor]:
        # TODO: Critic update
        pass

    def _train_actor(self, replay_data: ReplayBufferSamples) -> Dict[str, th.Tensor]:
        """
        Train the actor network and update the temperature and dual variables on the sampled data.
        """
        with th.no_grad():
            target_distributions = self.actor_target.predict_action_distribution(
                replay_data.next_observations
            ).proba_distribution
            next_action_samples = target_distributions.sample((self.num_samples,))

            tiled_observations = replay_data.observations.tile(self.num_samples)
            flat_observations = tiled_observations.reshape(self.num_samples, -1)  # Merge first two dims
            flat_actions = next_action_samples.reshape(self.num_samples, -1)  # Merge first two dims
            values = self.critic_target.q1_forward(
                flat_observations, flat_actions
            )  # Use q1_forward because MPO defaults to using only one critic
            values = values.reshape(self.num_samples, -1)

            target_distributions = Independent(target_distributions, -1)

        self.actor.optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        distributions = self.actor.predict_action_distribution(replay_data.observations).proba_distribution
        distributions = Independent(distributions, -1)

        temperature = F.softplus(self.log_temperature) + FLOAT_EPSILON
        alpha_mean = F.softplus(self.log_alpha_mean) + FLOAT_EPSILON
        alpha_std = F.softplus(self.log_alpha_std) + FLOAT_EPSILON
        weights, temperature_loss = weights_and_temperature_loss(values, self.epsilon, temperature)

        # Action penalization is quadratic beyond [-1, 1].
        if self.action_penalization:
            penalty_temperature = F.softplus(self.log_penalty_temperature) + FLOAT_EPSILON
            diff_bounds = next_action_samples - th.clamp(next_action_samples, -1, 1)
            action_bound_costs = -th.norm(diff_bounds, dim=-1)
            penalty_weights, penalty_temperature_loss = weights_and_temperature_loss(
                action_bound_costs, self.epsilon_penalty, penalty_temperature
            )
            weights += penalty_weights
            temperature_loss += penalty_temperature_loss

        # Decompose the policy into fixed-mean and fixed-std distributions
        fixed_std_distribution = Independent(Normal(distributions.mean, target_distributions.stddev), -1)
        fixed_mean_distribution = Independent(Normal(target_distributions.mean, distributions.stddev), -1)

        # Compute the decomposed policy losses.
        policy_mean_losses = (fixed_std_distribution.base_dist.log_prob(next_action_samples).sum(dim=-1) * weights).sum(dim=0)
        policy_mean_loss = -(policy_mean_losses).mean()
        policy_std_losses = (fixed_mean_distribution.base_dist.log_prob(next_action_samples).sum(dim=-1) * weights).sum(dim=0)
        policy_std_loss = -policy_std_losses.mean()

        # Compute the decomposed KL between the target and online policies.
        if self.per_dim_constraining:
            kl_mean = th.distributions.kl.kl_divergence(target_distributions.base_dist, fixed_std_distribution.base_dist)
            kl_std = th.distributions.kl.kl_divergence(target_distributions.base_dist, fixed_mean_distribution.base_dist)
        else:
            kl_mean = th.distributions.kl.kl_divergence(target_distributions, fixed_std_distribution)
            kl_std = th.distributions.kl.kl_divergence(target_distributions, fixed_mean_distribution)

        # Compute the alpha-weighted KL-penalty and dual losses.
        kl_mean_loss, alpha_mean_loss = parametric_kl_and_dual_losses(kl_mean, alpha_mean, self.epsilon_mean)
        kl_std_loss, alpha_std_loss = parametric_kl_and_dual_losses(kl_std, alpha_std, self.epsilon_std)

        # Combine losses.
        policy_loss = policy_mean_loss + policy_std_loss
        kl_loss = kl_mean_loss + kl_std_loss
        dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
        loss = policy_loss + kl_loss + dual_loss

        loss.backward()
        if self.gradient_clip > 0:
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
            th.nn.utils.clip_grad_norm_(self.dual_variables, self.gradient_clip)
        self.actor.optimizer.step()
        self.dual_optimizer.step()

        log_dict = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "dual_loss": dual_loss.item(),
            "loss": loss.item(),
            "temperature": temperature.mean().item(),
            "alpha_mean": alpha_mean.mean().item(),
            "alpha_std": alpha_std.mean().item(),
        }
        if self.action_penalization:
            log_dict["penalty_temperature"] = penalty_temperature.mean().item()

        return log_dict

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)

        critic_losses = []
        policy_losses, kl_losses, dual_losses, actor_losses = [], [], [], []
        logged_temperatures, logged_alpha_means, logged_alpha_stds = [], [], []
        if self.action_penalization:
            logged_penalties = []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            critic_log_dict = self._train_critic(replay_data)
            actor_log_dict = self._train_actor(replay_data)

            critic_losses.append(critic_log_dict["loss"])
            policy_losses.append(actor_log_dict["policy_loss"])
            kl_losses.append(actor_log_dict["kl_loss"])
            dual_losses.append(actor_log_dict["dual_loss"])
            actor_losses.append(actor_log_dict["loss"])
            logged_temperatures.append(actor_log_dict["temperature"])
            logged_alpha_means.append(actor_log_dict["alpha_mean"])
            logged_alpha_stds.append(actor_log_dict["alpha_std"])
            if self.action_penalization:
                logged_penalties.append(actor_log_dict["penalty_temperature"])

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/dual_loss", np.mean(dual_losses))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/temperature", np.mean(logged_temperatures))
        self.logger.record("train/alpha_mean", np.mean(logged_alpha_means))
        self.logger.record("train/alpha_std", np.mean(logged_alpha_stds))
        if self.action_penalization:
            self.logger.record("train/penalty_temperature", np.mean(logged_penalties))

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
        # TODO ?
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        # TODO ?
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []


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

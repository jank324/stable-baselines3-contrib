from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from torch.nn import functional as F

from sb3_contrib.mpo.policies import Actor, CnnPolicy, MlpPolicy, MPOPolicy, MultiInputPolicy

SelfMPO = TypeVar("SelfMPO", bound="MPO")


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
    policy: MPOPolicy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[MPOPolicy]],
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

        # TODO

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # TODO
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        # TODO ?
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # TODO Main todo

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):  # TODO: Are there multiple gradient steps in Acme? (I think there are higher up)
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # PSEUDO CODE (following Acme)
            # ----------------------
            # -> Get action distribution from policy based on based on "next observation" (o_t) from replay buffer
            # -> Get action distribution from target policy based on based on "next observation" (o_t) from replay buffer
            # -> Sample n actions from target action distribution ("to evaluate policy")
            # -> Compute the target critic's Q-value of the sampled actions in state o_t
            # -> Average target critic Q-values to compute the TD-learning bootstrap target (before this reshape to batch dims)
            # -> Compute online critic value of previous observation (o_t-1) and action (a_t-1) from replay buffer
            # -> Compute critic loss as TD(0) error (and average it)
            # -> Compute policy loss (through policy loss module from online and target action distributions, sampled actions
            #    and the latter's Q-values)
            #   MPO loss with decoupled KL constraints as in (Abdolmaleki et al., 2018)
            #   ----------------------
            #   -> Multivariate normal distribution of target and online action to independent normals
            #   -> Clip (logs) dual variables (temperature, alpha_mean and alpha_std) to (-18, None) to ensure they stay
            #      positive (are alphas Lagrange multipliers?)
            #   -> Transform dual variables from log to normal space using softplus instead of exp (for numerical stability)
            #      ... (and +1e-8 ... why?)
            #   -> Compute normalized importance weights, used to compute expectations with respect to the non-parametric
            #      policy; and the temperature loss, used to adapt the tempering of Q-values
            #      (computes_weights_and_temperature_loss function)
            #  (->) FOR DIAGNOSTICS: Compute estimated actualized KL between the non-parametric and current target policies
            #   -> IF PENALIZED_ACTIONS option is activated:
            #     -> SOME OTHER ADDITIONAL STUFF that adds to normalized weights and temperature loss
            #   -> Decompose the online policy into fixed-mean & fixed-stddev distributions (indepenent normals with target
            #      mean and online stddev, and vice versa) ... as per second paper works better
            #   -> Compute decomposed policy losses using each of the decomposed distributions (using passed actions,
            #      normalized weights and the respoective distribution)
            #   -> Compute the decomposed KL divergence between the target and online policies (KL divergences from target
            #      action distribution to either of the decombosed distributions) / (slightly different behaviour based on
            #      per_dim_constraining option)
            #   -> Compute the alpha-weighted KL-penalty and dual losses to adapt the alphas (loss_[kl|alpha]_[mean|std] using
            #      kl_[mean|std], alpha_[mean|std] and epsilon_[mean|std] ... epilson being a config option) ...
            #      (compute_parametric_kl_penalty_and_dual_loss function)
            #   -> Combine (sum up) losses policy and KL penalty mean and std ... dual loss made up of alpha mean and std, and
            #      temperature loss ... then all three summed (loss = loss_policy + loss_kl_penalty + loss_dual)
            #   ---------------------- (returns loss)
            # -> Clip gradients of critic and policy (and dual variables) ... (by global norm? 40.0?)
            # -> Apply optimisers to critic, policy and dual variables

            # PSUEDO CODE (following tonic) ... x means "no grad"
            # ----------------------
            # -> x Get target action distribution from target actor (from observations passed to updater (from replay
            #      buffer???))
            # -> x Sample n actions from target action distribution
            # -> x Compute the target critic's Q-value of the sampled actions using the passed observations
            # -> x Convert target action distribution from normal to independent normals to satisfy KL constraints
            #      per-dimension
            # -> Zero grad for actor optimiser dual optimiser
            # -> Get action distribution from (non-target) actor ... then convert to independent normals
            # -> Get dual variables (temperature, alpha_mean and alpha_std) from their logs using softplus and adding
            #    FLOAT_EPSILON = 1e-8
            # -> weights_and_temperature_loss function to compute weights and temperature loss from previously computed target
            #    critic values, self.episolon and temperature
            # -> IF ACTION_PENALIZATION option is activated:
            #   -> SOME OTHER ADDITIONAL STUFF
            # -> Decompose the policy (target action distribution from earlier) into fixed-mean and fixed-std distributions
            #    mean from non-target action distribution and std from target action distribution, and vice versa (as
            #    independent normals)
            # -> Compute the decomposed policy losses (policy_mean_loss and policy_std_loss)
            # -> Compute the decomposed KL divergences between the target and online policies (kl_mean and kl_std) ... slighty
            #    different behaviour based on per_dim_constraining option
            # -> Compute the alpha-weighted KL-penalty and dual losses (kl_[mean|std]_loss and alpha_[mean|std]_loss) ... based
            #    on kl_[mean|std], alpha_[mean|std] and epsilon_[mean|std] ... epsilon being a config option
            # -> Combine (sum up) losses:
            #      - policy loss (policy_mean_loss + policy_std_loss)
            #      - KL (penalty) loss (kl_mean_loss + kl_std_loss)
            #      - dual loss (alpha_mean_loss + alpha_std_loss + temperature_loss)
            #      -----------------------------------------
            #      - total loss (policy_loss + kl_loss + dual_loss)
            # -> Backpropagate loss ... and IF gradient_clip (OPTION?) is True: clip_grad_norm for actor variables and dual
            #    varialbes
            # -> Step optimisers (actor optimiser and dual optimiser)

            with th.no_grad():
                # TODO: Get action distribution from policy based on based on "next observation" from replay buffer
                next_action_distribution = self.actor(replay_data.next_observations)
                # TODO: Get action distribution from target policy based on based on "next observation" from replay buffer
                next_action_distribution_target = self.actor_target(replay_data.next_observations)

                # ----------------------

                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # TODO: Is it Polyak update in Acme or just a copy?
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

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

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name

from sb3_contrib.mpo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

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

        # TODO: ?

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # TODO Main todo

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)

        policy_losses, kl_losses, dual_losses, losses = [], [], [], []
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
                # next_actions, next_action_values, next_log_prob, next_action_distribution = self.policy_target.forward(
                #     replay_data.next_observations
                # )
                next_action_distributions = self.actor_target.get_distribution(replay_data.next_observations)
                next_action_samples = next_action_distributions.sample(self.num_samples)

                tiled_observations = updaters.tile(observations, self.num_samples)
                flat_observations = updaters.merge_first_two_dims(tiled_observations)
                flat_actions = updaters.merge_first_two_dims(actions)
                values = self.model.target_critic(flat_observations, flat_actions)
                values = values.view(self.num_samples, -1)

                assert isinstance(target_distributions, th.distributions.normal.Normal)
                target_distributions = independent_normals(target_distributions)

            self.actor_optimizer.zero_grad()
            self.dual_optimizer.zero_grad()

            distributions = self.model.actor(observations)
            distributions = independent_normals(distributions)

            temperature = th.nn.functional.softplus(self.log_temperature) + FLOAT_EPSILON
            alpha_mean = th.nn.functional.softplus(self.log_alpha_mean) + FLOAT_EPSILON
            alpha_std = th.nn.functional.softplus(self.log_alpha_std) + FLOAT_EPSILON
            weights, temperature_loss = weights_and_temperature_loss(values, self.epsilon, temperature)

            # Action penalization is quadratic beyond [-1, 1].
            if self.action_penalization:
                penalty_temperature = th.nn.functional.softplus(self.log_penalty_temperature) + FLOAT_EPSILON
                diff_bounds = actions - th.clamp(actions, -1, 1)
                action_bound_costs = -th.norm(diff_bounds, dim=-1)
                penalty_weights, penalty_temperature_loss = weights_and_temperature_loss(
                    action_bound_costs, self.epsilon_penalty, penalty_temperature
                )
                weights += penalty_weights
                temperature_loss += penalty_temperature_loss

            # Decompose the policy into fixed-mean and fixed-std distributions.
            fixed_std_distribution = independent_normals(distributions.base_dist, target_distributions.base_dist)
            fixed_mean_distribution = independent_normals(target_distributions.base_dist, distributions.base_dist)

            # Compute the decomposed policy losses.
            policy_mean_losses = (fixed_std_distribution.base_dist.log_prob(actions).sum(dim=-1) * weights).sum(dim=0)
            policy_mean_loss = -(policy_mean_losses).mean()
            policy_std_losses = (fixed_mean_distribution.base_dist.log_prob(actions).sum(dim=-1) * weights).sum(dim=0)
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
                th.nn.utils.clip_grad_norm_(self.actor_variables, self.gradient_clip)
                th.nn.utils.clip_grad_norm_(self.dual_variables, self.gradient_clip)
            self.actor_optimizer.step()
            self.dual_optimizer.step()

            dual_variables = dict(
                temperature=temperature.detach(), alpha_mean=alpha_mean.detach(), alpha_std=alpha_std.detach()
            )
            if self.action_penalization:
                dual_variables["penalty_temperature"] = penalty_temperature.detach()

            # Save losses
            policy_losses.append(policy_loss.item())
            kl_losses.append(kl_loss.item())
            dual_losses.append(dual_loss.item())
            losses.append(loss.item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/dual_loss", np.mean(dual_losses))
        self.logger.record("train/loss", np.mean(losses))

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

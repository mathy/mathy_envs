# Requies: torch torchvision torchinfo mathy_envs gymnasium tqdm matplotlib numpy
#
# Based on: https://github.com/nikhilbarhate99/PPO-PyTorch
#
# @nikhilbarhate99 🙇 - https://github.com/nikhilbarhate99
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
from torchinfo import summary

import mathy_envs.gym  # noqa


@dataclass
class PPOConfig:
    env_names: List[str]  # Environment name
    state_dim: int  # Dimension of the state space
    random_seed: int  # set random seed if required (0 = no random seed)
    action_dim: int  # Dimension of the action space
    lr_actor: float  # Learning rate for the actor network
    lr_critic: float  # Learning rate for the critic network
    gamma: float  # Discount factor
    K_epochs: int  # Number of epochs to update the policy
    eps_clip: float  # Clip parameter for PPO
    has_continuous_action_space: bool  # Whether the action space is continuous or discrete
    device: torch.device  # Device to run the training on
    action_std_init: float = (
        0.6  # Initial standard deviation for the action distribution
    )
    critic_hidden_dim: int = 64  # Dimension of the hidden layer in the critic network


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        has_continuous_action_space: bool,
        action_std_init: float,
        device: torch.device,
    ):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device
        self.action_dim = action_dim

        if has_continuous_action_space:
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init
            ).to(self.device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1),
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std * new_action_std
            ).to(self.device)
        else:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print(
                "WARNING : Calling ActorCritic::set_action_std() on discrete action space policy"
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            use_mask = True
            if use_mask:
                mask = state[-action_probs.shape[0] :]
                action_probs = action_probs * mask
                action_probs = action_probs / torch.sum(action_probs)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma,
        K_epochs: int,
        eps_clip,
        has_continuous_action_space: bool,
        device: torch.device,
        action_std_init: float = 0.6,
        critic_hidden_dim: int = 64,
    ):
        self.has_continuous_action_space = has_continuous_action_space
        self.device = device

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            device=device,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=critic_hidden_dim,
            has_continuous_action_space=has_continuous_action_space,
            action_std_init=action_std_init,
            device=device,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print(
                "--------------------------------------------------------------------------------------------"
            )
            print(
                "WARNING : Calling PPO::set_action_std() on discrete action space policy"
            )
            print(
                "--------------------------------------------------------------------------------------------"
            )

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print(
            "--------------------------------------------------------------------------------------------"
        )

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print(
                    "setting actor output action_std to min_action_std : ",
                    self.action_std,
                )
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print(
                "WARNING : Calling PPO::decay_action_std() on discrete action space policy"
            )

        print(
            "--------------------------------------------------------------------------------------------"
        )

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )


################################### Training ###################################
def train(config: PPOConfig, init_from: Optional[str] = None) -> None:
    print(f"Device set to: {torch.cuda.get_device_name(config.device)}")
    print(
        "============================================================================================"
    )
    print(f"training environments : {', '.join(config.env_names)}")

    ####### initialize environment hyperparameters ######

    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(
        50e6
    )  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = (
        0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    )
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    #####################################################

    envs = [
        gym.make(name, invalid_action_response="raise", verbose=False)
        for name in config.env_names
    ]
    assert len(envs) > 0, "No environments found"
    env = envs[0]

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if config.has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    env_short = config.env_names[0] if len(config.env_names) == 1 else "multi"
    root_path = Path("./trained") / env_short
    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    log_dir = root_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = f"{log_dir}/log_{run_num}.csv"
    print(f"current logging run number for {env_short}: {run_num}")
    print(f"logging at: {log_f_name}")
    #####################################################

    ################### checkpointing ###################
    checkpoint_path = root_path / "model.pth"
    print(f"save checkpoint path: {checkpoint_path}")
    #####################################################

    ############# print all hyperparameters #############
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print(f"max training timesteps: {max_training_timesteps}")
    print(f"max timesteps per episode: {max_ep_len}")
    print(f"model saving frequency: {save_model_freq} timesteps")
    print(f"log frequency: {log_freq} timesteps")
    print(f"printing average reward over episodes in last: {print_freq} timesteps")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print(f"state space dimension: {state_dim}")
    print(f"action space dimension: {action_dim}")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    if config.has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print(f"starting std of action distribution: {action_std}")
        print(f"decay rate of std of action distribution: {action_std_decay_rate}")
        print(f"minimum std of action distribution: {min_action_std}")
        print(
            f"decay frequency of std of action distribution: {action_std_decay_freq} timesteps"
        )
    else:
        print("Initializing a discrete action space policy")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print(f"PPO update frequency: {update_timestep} timesteps")
    print(f"PPO K epochs: {config.K_epochs}")
    print(f"PPO epsilon clip: {config.eps_clip}")
    print(f"discount factor (gamma): {config.gamma}")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print(f"optimizer learning rate actor: {config.lr_actor}")
    print(f"optimizer learning rate critic: {config.lr_critic}")
    if config.random_seed:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print(f"setting random seed to {config.random_seed}")
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    #####################################################

    print(
        "============================================================================================"
    )

    print(f"max training timesteps : {max_training_timesteps}")
    print(f"max timesteps per episode : {max_ep_len}")
    print(f"model saving frequency : {save_model_freq} timesteps")
    print(f"log frequency : {log_freq} timesteps")
    print(f"printing average reward over episodes in last : {print_freq} timesteps")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print(f"state space dimension : {state_dim}")
    print(f"action space dimension : {action_dim}")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    if config.has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print(f"starting std of action distribution : {action_std}")
        print(f"decay rate of std of action distribution : {action_std_decay_rate}")
        print(f"minimum std of action distribution : {min_action_std}")
        print(
            f"decay frequency of std of action distribution : {action_std_decay_freq} timesteps"
        )
    else:
        print("Initializing a discrete action space policy")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print(f"PPO update frequency : {update_timestep} timesteps")
    print(f"PPO K epochs : {config.K_epochs}")
    print(f"PPO epsilon clip : {config.eps_clip}")
    print(f"discount factor (gamma) : {config.gamma}")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print(f"optimizer learning rate actor : {config.lr_actor}")
    print(f"optimizer learning rate critic : {config.lr_critic}")
    if config.random_seed:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print(f"setting random seed to :{config.random_seed}")
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    #####################################################

    print(
        "============================================================================================"
    )

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(
        state_dim,
        action_dim,
        config.lr_actor,
        config.lr_critic,
        config.gamma,
        config.K_epochs,
        config.eps_clip,
        config.has_continuous_action_space,
        config.device,
        action_std,
        critic_hidden_dim=config.critic_hidden_dim,
    )
    summary(ppo_agent.policy.actor, input_size=(state_dim,))
    summary(ppo_agent.policy.critic, input_size=(state_dim,))

    if init_from is not None:
        print(f"loading network from : {init_from}")
        ppo_agent.load(init_from)
        print(
            "--------------------------------------------------------------------------------------------"
        )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print(f"Started training at (GMT) : {start_time}")

    print(
        "============================================================================================",
        flush=True,
    )

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        # Choose a random env
        env = np.random.choice(envs)

        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += float(reward)

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if (
                config.has_continuous_action_space
                and time_step % action_std_decay_freq == 0
            ):
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(
                    f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t Average Reward : {print_avg_reward}",
                    flush=True,
                )

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                print(f"saving model at : {checkpoint_path}")
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print(
                    f"Elapsed Time  : {datetime.now().replace(microsecond=0) - start_time}"
                )
                print(
                    "--------------------------------------------------------------------------------------------",
                    flush=True,
                )

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================"
    )


def test(config: PPOConfig, checkpoint_path: str):
    print(
        "============================================================================================"
    )

    envs = [
        gym.make(name, invalid_action_response="raise", verbose=True)
        for name in config.env_names
    ]
    assert len(envs) > 0, "No environments found"
    env = envs[0]

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if config.has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(
        state_dim,
        action_dim,
        config.lr_actor,
        config.lr_critic,
        config.gamma,
        config.K_epochs,
        config.eps_clip,
        config.has_continuous_action_space,
        config.device,
    )

    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    print(
        "--------------------------------------------------------------------------------------------"
    )

    test_running_reward = 0
    total_test_episodes = 1000  # total num of testing episodes
    for ep in range(1, total_test_episodes + 1):
        env = np.random.choice(envs)
        ep_reward = 0
        print(env.mathy.get_env_namespace())
        state, _ = env.reset()
        done = False
        while not done:
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            # time.sleep(0.15)
            done = terminated or truncated
            ep_reward += float(reward)

        # clear buffer
        ppo_agent.buffer.clear()
        test_running_reward += ep_reward

        emoji = "✅" if ep_reward >= 1.3 else "🟨" if ep_reward >= 0.6 else "🔴"
        spacer = "=" * 100
        print(f"{ep} {spacer} {emoji} Reward: {round(ep_reward, 2)}")
        ep_reward = 0

    env.close()

    print(
        "============================================================================================"
    )

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print(
        "============================================================================================"
    )


EnvTypes = Literal[
    "poly",
    "poly-blockers",
    "poly-combine",
    "poly-commute",
    "poly-grouping",
    "poly-like-terms-haystack",
    "binomial",
    "complex",
]

if __name__ == "__main__":
    env_difficulty: Literal["easy", "normal", "hard"] = "easy"
    env_names = [
        f"mathy-{t}-{env_difficulty}-v0"
        for t in [
            "poly",
            "poly-blockers",
            "poly-combine",
            "poly-commute",
            "poly-grouping",
            "poly-like-terms-haystack",
            "binomial",
            "complex",
        ]
    ]
    config = PPOConfig(
        env_names=env_names,
        has_continuous_action_space=False,  # continuous action space; else discrete
        random_seed=1337,  # set random seed if required (0 = no random seed)
        state_dim=0,
        action_dim=0,
        lr_actor=0.0003,  # learning rate for actor network
        lr_critic=0.001,  # learning rate for critic network
        gamma=0.99,  # discount factor
        K_epochs=80,  # update policy for K epochs in one PPO update
        eps_clip=0.2,  # clip parameter for PPO
        device=torch.device("cpu" if not torch.cuda.is_available() else "cuda:0"),
    )
    if len(sys.argv) == 2:
        test(config, sys.argv[1])
    else:
        train(config, sys.argv[1] if len(sys.argv) > 1 else None)

import sys
from typing import Any, cast

import mock
import numpy as np
import pytest

from mathy_envs.env import MathyEnv
from mathy_envs.state import MathyObservation


def test_gym_raises_helpful_error_if_gym_is_not_installed():
    with mock.patch.dict(sys.modules, {"gymnasium": None}):
        with pytest.raises(ImportError, match=r"pip install mathy_envs\[gym\]"):
            import mathy_envs.gym  # noqa


def test_gym_instantiate_envs():
    import gymnasium as gym

    from mathy_envs.gym import MathyGymEnv

    all_envs = gym.registry.values()
    # Filter to just mathy registered envs
    mathy_gym_envs = [e for e in all_envs if e.id.startswith("mathy-")]

    assert len(mathy_gym_envs) > 0

    # Each env can be created and produce an initial observation without
    # special configuration.
    for gym_env_spec in mathy_gym_envs:
        print(f"Testing {gym_env_spec.id}...")
        wrapper_env: MathyGymEnv = gym.make(gym_env_spec.id)  # type:ignore
        assert wrapper_env is not None

        obs = wrapper_env.reset()
        print("initial observation:", obs)
        action = wrapper_env.action_space.sample()
        wrapper_env.step(action)
        observation, info = wrapper_env.reset()
        assert isinstance(observation, (dict, np.ndarray, MathyObservation))
        assert observation is not None


def test_gym_step_info():
    import gymnasium as gym

    from mathy_envs.gym import MathyGymEnv

    all_envs = gym.registry.values()
    # Filter to just mathy registered envs
    mathy_gym_envs = [e for e in all_envs if e.id.startswith("mathy-")]

    assert len(mathy_gym_envs) > 0

    # Each env can be created and produce an initial observation without
    # special configuration.
    for gym_env_spec in mathy_gym_envs:
        print(f"Testing {gym_env_spec.id}...")
        wrapper_env: MathyGymEnv = gym.make(gym_env_spec.id)  # type:ignore
        assert wrapper_env is not None

        obs = wrapper_env.reset()
        print("initial observation:", obs)
        action = wrapper_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapper_env.step(action)
        for key in ["valid", "done", "truncated", "transition"]:
            assert key in info, f"info should have a string key {key}"


def test_gym_env_spaces():
    import gymnasium as gym

    from mathy_envs.gym import MaskedDiscrete, MathyGymEnv

    wrapper_env: MathyGymEnv = gym.make("mathy-poly-easy-v0").unwrapped  # type:ignore
    mathy: MathyEnv = wrapper_env.mathy
    observation, info = wrapper_env.reset()

    # Has a masked discrete (finite) action space
    assert wrapper_env.action_space is not None
    assert isinstance(wrapper_env.action_space, MaskedDiscrete)
    # Action space is (num_rules * max_seq_len)
    assert wrapper_env.action_size == int(len(mathy.rules) * mathy.max_seq_len)
    assert wrapper_env.action_space.shape == tuple()

    # Observation matches the space spec
    assert wrapper_env.observation_space.shape == cast(Any, observation).shape


def test_gym_probability_action_mask():
    import gymnasium as gym

    from mathy_envs.gym import MathyGymEnv

    env: MathyGymEnv = gym.make(  # type:ignore
        "mathy-poly-easy-v0", mask_as_probabilities=True
    )
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    offset = -env.unwrapped.mathy.action_size  # type:ignore
    action_mask = np.array(obs[offset:])
    # When returned as probabilities, the mask sums to almost 1.0
    assert np.sum(action_mask) > 1.0 - 1e-4

    env: MathyGymEnv = gym.make(  # type:ignore
        "mathy-poly-easy-v0", mask_as_probabilities=False
    )
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    # When not as probabilities, the mask is a bunch of 1s and 0s
    action_mask = np.array(obs[offset:])
    assert np.sum(action_mask) > 1

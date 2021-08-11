import sys

import mock
import numpy as np
import pytest

from mathy_envs.env import MathyEnv
from mathy_envs.state import MathyObservation


def test_gym_raises_helpful_error_if_gym_is_not_installed():
    with mock.patch.dict(sys.modules, {"gym": None}):
        with pytest.raises(ImportError, match=r"pip install mathy_envs\[gym\]"):
            import mathy_envs.gym  # noqa


def test_gym_instantiate_envs():
    import gym

    from mathy_envs.gym import MathyGymEnv

    all_envs = gym.envs.registration.registry.all()  # type:ignore
    # Filter to just mathy registered envs
    mathy_gym_envs = [e for e in all_envs if e.id.startswith("mathy-")]

    assert len(mathy_gym_envs) > 0

    # Each env can be created and produce an initial observation without
    # special configuration.
    for gym_env_spec in mathy_gym_envs:
        wrapper_env: MathyGymEnv = gym.make(gym_env_spec.id)  # type:ignore
        assert wrapper_env is not None

        obs = wrapper_env.reset()
        print("initial observation:", obs)
        action = wrapper_env.action_space.sample()
        wrapper_env.step(action)
        observation: np.ndarray = wrapper_env.reset()
        assert isinstance(observation, (dict, np.ndarray, MathyObservation))
        assert observation is not None


def test_gym_env_spaces():
    import gym

    from mathy_envs.gym import MaskedDiscrete, MathyGymEnv

    wrapper_env: MathyGymEnv = gym.make("mathy-poly-easy-v0")  # type:ignore
    mathy: MathyEnv = wrapper_env.mathy
    observation: np.ndarray = wrapper_env.reset()

    # Has a masked discrete (finite) action space
    assert wrapper_env.action_space is not None
    assert isinstance(wrapper_env.action_space, MaskedDiscrete)
    # Action space is (num_rules * max_seq_len)
    assert wrapper_env.action_size == int(len(mathy.rules) * mathy.max_seq_len)
    assert wrapper_env.action_space.shape == tuple()

    # Observation matches the space spec
    assert wrapper_env.observation_space.shape == observation.shape


def test_gym_probability_action_mask():
    import gym

    from mathy_envs.gym import MathyGymEnv

    env: MathyGymEnv = gym.make(
        "mathy-poly-easy-v0", mask_as_probabilities=True
    )  # type:ignore
    obs = env.reset()
    offset = -env.mathy.action_size
    action_mask = obs[offset:]
    # When returned as probabilities, the mask sums to 1.0
    assert np.sum(action_mask) == 1.0

    env: MathyGymEnv = gym.make(
        "mathy-poly-easy-v0", mask_as_probabilities=False
    )  # type:ignore
    obs = env.reset()
    # When not as probabilities, the mask is a bunch of 1s and 0s
    action_mask = obs[offset:]
    assert np.sum(action_mask) > 1

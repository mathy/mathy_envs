import sys
from mock import Mock

import pytest

import mock


def test_gym_raises_helpful_error_if_gym_is_not_installed():

    with mock.patch.dict(sys.modules, {"gym": None}):
        with pytest.raises(ImportError, match=r"pip install mathy_envs\[gym\]"):
            import mathy_envs.gym

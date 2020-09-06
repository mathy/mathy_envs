import sys

import mock
import pytest


def test_gym_raises_helpful_error_if_gym_is_not_installed():

    with mock.patch.dict(sys.modules, {"gym": None}):
        with pytest.raises(ImportError, match=r"pip install mathy_envs\[gym\]"):
            pass

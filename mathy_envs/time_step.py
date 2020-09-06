# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Changes:
#
#  - 2019/12/09 JD: inline and remove tensorflow dependency/shape support
#  - 2019/12/11 JD: add MathyObservation type hints
#  - 2020/04/04 JD: remove more unused code
"""TimeStep representing a step in the environment.

This file is a mostly direct copy of the implementation from the
[tf_agents](https://github.com/tensorflow/agents) library but has
the dependency on tensorflow removed along with advanced shape
features.

Mathy doesn't use these features and the overhead of loading tensorflow
to pass environment states around is not great for things like CLI start
times.
"""
import collections

import numpy

from mathy_envs.state import MathyObservation


class TimeStep(
    collections.namedtuple(
        "TimeStep", ["step_type", "reward", "discount", "observation"]
    )
):
    __slots__ = ()

    def __hash__(self) -> int:
        return hash(tuple(self))


class StepType(object):
    """Defines the status of a `TimeStep` within a sequence."""

    # Denotes the first `TimeStep` in a sequence.
    FIRST = numpy.array([0], dtype="int32")
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = numpy.array([1], dtype="int32")
    # Denotes the last `TimeStep` in a sequence.
    LAST = numpy.array([2], dtype="int32")


def transition(
    observation: MathyObservation, reward: float, discount: float = 1.0
) -> TimeStep:
    """Returns a `TimeStep` with `step_type` set equal to `StepType.MID`."""
    return TimeStep(StepType.MID, reward, discount, observation)


def termination(observation: MathyObservation, reward: float) -> TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
    return TimeStep(StepType.LAST, reward, 00, observation)


def is_terminal_transition(transition: TimeStep) -> bool:
    return bool(transition.step_type == StepType.LAST)

import random
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from mathy_core import STOP
from mathy_core.expressions import MathExpression
from mathy_core.parser import ExpressionParser, InvalidSyntax
from mathy_core.rule import BaseRule, ExpressionChangeRule
from mathy_core.rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    RestateSubtractionRule,
    VariableMultiplyRule,
)
from mathy_core.tree import BinaryTreeNode, VisitStop
from mathy_core.util import compare_expression_string_values, raise_with_history

from . import time_step
from .state import (
    MathyEnvState,
    MathyEnvStateStep,
    MathyObservationUnion,
    ObservationType,
)
from .time_step import is_terminal_transition
from .types import ActionType, EnvRewards, Literal, MathyEnvProblem, MathyEnvProblemArgs

InvalidActionResponses = Literal["raise", "penalize", "terminal"]

INVALID_ACTION_RESPONSES: List[InvalidActionResponses] = [
    "raise",
    "penalize",
    "terminal",
]


class MathyEnv:
    """Implement a math solving game where a player wins by executing the
    right sequence of actions to reduce a math expression to an agreeable
    basic representation in as few moves as possible."""

    rules: List[BaseRule]
    max_moves: int
    max_seq_len: int
    verbose: bool
    reward_discount: float
    parser: ExpressionParser
    valid_actions_mask_cache: Dict[str, List[List[int]]]
    valid_rules_cache: Dict[str, List[int]]
    invalid_action_response: InvalidActionResponses
    previous_state_penalty: bool
    preferred_term_commute: bool

    def __init__(
        self,
        *,
        rules: Optional[List[BaseRule]] = None,
        max_moves: int = 20,
        verbose: bool = False,
        invalid_action_response: InvalidActionResponses = "raise",
        reward_discount: float = 0.99,
        max_seq_len: int = 128,
        previous_state_penalty: bool = True,
        preferred_term_commute: bool = False,
    ):
        self.discount = reward_discount
        self.previous_state_penalty = previous_state_penalty
        self.verbose = verbose
        self.max_moves = max_moves
        self.max_seq_len = max_seq_len
        self.invalid_action_response = invalid_action_response
        self.parser = ExpressionParser()
        if rules is None:
            self.rules = MathyEnv.core_rules(
                preferred_term_commute=preferred_term_commute
            )
        else:
            self.rules = rules
        self.valid_actions_mask_cache = dict()
        self.valid_rules_cache = dict()

        if self.invalid_action_response not in INVALID_ACTION_RESPONSES:
            raise ValueError(
                f"Unknown invalid action behavior: {self.invalid_action_response}\n"
                f"Expected one of: {', '.join(INVALID_ACTION_RESPONSES)}"
            )

    @classmethod
    def core_rules(cls, preferred_term_commute: bool = False) -> List[BaseRule]:
        """Return the mathy core agent actions"""
        return [
            ConstantsSimplifyRule(),
            CommutativeSwapRule(preferred=preferred_term_commute),
            DistributiveMultiplyRule(),
            DistributiveFactorOutRule(),
            AssociativeSwapRule(),
            VariableMultiplyRule(),
            RestateSubtractionRule(),
        ]

    @property
    def action_size(self) -> int:
        """Return the number of available actions"""
        return len(self.rules) * self.max_seq_len

    def finalize_state(self, state: MathyEnvState) -> None:
        """Perform final checks on a problem state, to ensure the episode yielded
        results that were uncorrupted by transformation errors."""
        from_timestep: MathyEnvStateStep = state.agent.history[0]
        to_timestep: MathyEnvStateStep = state.agent.history[-1]
        compare_expression_string_values(
            str(from_timestep.raw), str(to_timestep.raw), state.agent.history
        )

    def get_env_namespace(self) -> str:
        """Return a unique dot namespaced string representing the current
        environment. e.g. mycompany.envs.differentiate"""
        raise NotImplementedError("subclass must implement this")

    def get_rewarding_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        """Get the list of rewarding action types. When these actions
        are selected, the agent gets a positive reward."""
        # NOTE: by default we give a positive reward for most actions taken. Reward
        #       values are only applied AFTER penalties, so things like reentrant
        #       states become negative reward even if their action is otherwise
        #       rewarding.
        return [
            RestateSubtractionRule,
            ConstantsSimplifyRule,
            DistributiveFactorOutRule,
            VariableMultiplyRule,
        ]

    def get_penalizing_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        """Get the list of penalizing action types. When these actions
        are selected, the agent gets a negative reward."""
        return [
            AssociativeSwapRule,
            DistributiveMultiplyRule,
        ]

    def max_moves_fn(
        self, problem: MathyEnvProblem, config: MathyEnvProblemArgs
    ) -> int:
        """Return the environment specific maximum move count for a given prolem."""
        return problem.complexity * 3

    def transition_fn(
        self,
        env_state: MathyEnvState,
        expression: MathExpression,
        features: MathyObservationUnion,
    ) -> Optional[time_step.TimeStep]:
        """Provide environment-specific transitions per timestep."""
        return None

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        """Return a problem for the environment given a set of parameters
        to control problem generation.

        This is implemented per environment so each environment can
        generate its own dataset with no required configuration."""
        raise NotImplementedError("This must be implemented in a subclass")

    def state_to_observation(
        self,
        state: MathyEnvState,
        max_seq_len: Optional[int] = None,
        obs_type: ObservationType = ObservationType.FLAT,
    ) -> MathyObservationUnion:
        """Convert an environment state into an observation that can be used
        by a training agent."""

        action_mask = self.get_valid_moves(state)
        observation = state.to_observation(
            move_mask=action_mask,
            parser=self.parser,
            max_seq_len=max_seq_len,
            obs_type=obs_type,
        )
        return observation

    def get_win_signal(self, env_state: MathyEnvState) -> float:
        """Calculate the reward value for completing the episode. This is done
        so that the reward signal can be scaled based on the time it took to
        complete the episode."""
        tiny = 3e-10
        total_moves = max(tiny, env_state.max_moves)
        # guard against divide by zero with max and a small value
        current_move = max(tiny, total_moves - env_state.agent.moves_remaining)
        bonus = (total_moves / current_move) / total_moves
        # If the episode is not very short, and the agent completes in half
        # the number of allowed steps, double the bonus signal
        if total_moves > 10 and current_move < total_moves / 2:
            bonus *= 2
        return min(2.0, EnvRewards.WIN + bonus)

    def get_lose_signal(self, env_state: MathyEnvState) -> float:
        """Calculate the reward value for failing to complete the episode. This is done
        so that the reward signal can be problem-type dependent."""
        return EnvRewards.LOSE

    def get_state_transition(self, env_state: MathyEnvState) -> time_step.TimeStep:
        """Given an input state calculate the transition value of the timestep.

        # Parameters
            env_state: current env_state

        # Returns
            transition: the current state value transition
        """
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        features = env_state.to_observation(
            self.get_valid_moves(env_state), parser=self.parser
        )
        root = expression.get_root()
        assert isinstance(root, MathExpression)

        # Subclass specific win conditions happen here. Custom win-conditions
        # outside of that can override this method entirely.
        result = self.transition_fn(env_state, root, features)
        if result is not None:
            return result

        # Check the turn count last because if the previous move that incremented
        # the turn over the count resulted in a win-condition, we want it to be honored.
        if env_state.agent.moves_remaining <= 0:
            return time_step.termination(features, self.get_lose_signal(env_state))

        # The agent is penalized for returning to a previous state.
        if self.previous_state_penalty is True:
            for key, group in groupby(
                sorted([f"{h.raw}" for h in env_state.agent.history])
            ):
                list_count = len(list(group))
                if list_count <= 1 or key != expression.raw:
                    continue

                # After more than (n) visits to the same state, you lose.
                if list_count > 3:
                    return time_step.termination(
                        features, self.get_lose_signal(env_state)
                    )

                # NOTE: the reward is scaled by # of times this state has been visited
                return time_step.transition(
                    features,
                    reward=EnvRewards.PREVIOUS_LOCATION * list_count,
                    discount=self.discount,
                )

        if len(agent.history) > 0:
            last_timestep = agent.history[-1]
            rule = self.get_rule_from_timestep(last_timestep)
            reward_actions = self.get_rewarding_actions(env_state)
            # The rewarding_actions can be user specified
            for rewarding_class in reward_actions:
                if isinstance(rule, rewarding_class):
                    return time_step.transition(
                        features,
                        reward=EnvRewards.HELPFUL_MOVE,
                        discount=self.discount,
                    )

            penalty_actions = self.get_penalizing_actions(env_state)
            # The rewarding_actions can be user specified
            for penalty_class in penalty_actions:
                if isinstance(rule, penalty_class):
                    return time_step.transition(
                        features,
                        reward=EnvRewards.UNHELPFUL_MOVE,
                        discount=self.discount,
                    )

        # We're in a new state, and the agent is a little older.
        return time_step.transition(
            features, reward=EnvRewards.TIMESTEP, discount=self.discount
        )

    def get_next_state(
        self, env_state: MathyEnvState, action: Union[int, np.int64, ActionType]
    ) -> Tuple[MathyEnvState, time_step.TimeStep, ExpressionChangeRule]:
        """
        # Parameters
        env_state: current env_state
        action:    a tuple of two integers representing the rule and node to act on

        # Returns
        next_state: env_state after applying action

        transition: the timestep that represents the state transition

        change: the change descriptor describing the change that happened
        """
        action = self.to_action(action)
        agent = env_state.agent
        expression = self.parser.parse(agent.problem)
        assert isinstance(
            action, (tuple, list)
        ), f"Expected tuple action, but received: {type(action)} {action}"
        action_index, token_index = action
        token = self.get_token_at_index(expression, token_index)
        operation = self.rules[action_index]

        op_not_rule = not isinstance(operation, BaseRule)
        op_cannot_apply = token is None or operation.can_apply_to(token) is False
        if token is None or op_not_rule or op_cannot_apply:
            if self.invalid_action_response == "raise":
                steps = int(env_state.max_moves - agent.moves_remaining)
                msg = "Step: {} - Invalid action({}) '{}' for expression '{}'.".format(
                    steps, action, type(operation), expression
                )
                raise_with_history("Invalid Action", msg, agent.history)
            elif self.invalid_action_response == "penalize":
                #
                out_env = env_state.get_out_state(
                    problem=env_state.agent.problem,
                    action=action,
                    moves_remaining=agent.moves_remaining - 1,
                )
                obs = out_env.to_observation(
                    self.get_valid_moves(out_env), parser=self.parser
                )
                transition = time_step.transition(obs, EnvRewards.INVALID_MOVE)
                return out_env, transition, ExpressionChangeRule(BaseRule())
            elif self.invalid_action_response == "terminal":
                out_env = env_state.get_out_state(
                    problem=env_state.agent.problem,
                    action=action,
                    moves_remaining=0,
                )
                obs = out_env.to_observation(
                    self.get_valid_moves(out_env), parser=self.parser
                )
                transition = time_step.termination(obs, self.get_lose_signal(env_state))
                return out_env, transition, ExpressionChangeRule(BaseRule())

        assert token is not None
        change = operation.apply_to(token.clone_from_root())
        assert change.result is not None
        root = change.result.get_root()
        change_name = operation.name
        out_problem = str(root)
        out_env = env_state.get_out_state(
            problem=out_problem,
            action=action,
            moves_remaining=agent.moves_remaining - 1,
        )

        transition = self.get_state_transition(out_env)
        if self.verbose:
            token_idx = int("{}".format(token_index).zfill(3))
            self.print_state(
                out_env, change_name[:25].lower(), token_idx, change, transition.reward
            )
        return out_env, transition, change

    def print_state(
        self,
        env_state: MathyEnvState,
        action_name: str,
        token_index: int = -1,
        change: Optional[ExpressionChangeRule] = None,
        change_reward: float = 0.0,
        pretty: bool = False,
    ) -> None:
        """Render the given state to stdout for visualization"""
        print(
            self.render_state(
                env_state, action_name, token_index, change, change_reward, pretty
            ),
            flush=True,
        )

    def is_terminal_state(self, env_state: MathyEnvState) -> bool:
        """Determine if a given state is terminal or not.

        # Arguments
        env_state (MathyEnvState): The state to inspect

        # Returns
        (bool): A boolean indicating if the state is terminal or not.
        """
        return is_terminal_transition(self.get_state_transition(env_state))

    def print_history(self, env_state: MathyEnvState, pretty: bool = True) -> None:
        """Render the history of an episode from a given state.

        # Arguments
        env_state (MathyEnvState): The state to render the history of.
        """
        history: List[MathyEnvStateStep] = env_state.agent.history[:]
        initial_step: MathyEnvStateStep = history.pop(0)
        curr_state: MathyEnvState = MathyEnvState(
            problem=initial_step.raw,
            max_moves=env_state.max_moves,
        )
        self.print_state(curr_state, "initial-state", pretty=pretty)
        while len(history) > 0:
            step: MathyEnvStateStep = history.pop(0)
            curr_state, transition, change = self.get_next_state(
                curr_state, step.action
            )
            rule_idx, token_idx = step.action
            rule: BaseRule = self.rules[rule_idx]
            rule_name: str = rule.name[:25].lower()
            self.print_state(
                pretty=pretty,
                env_state=curr_state,
                action_name=rule_name,
                token_index=int(f"{token_idx}".zfill(3)),
                change=change,
                change_reward=transition.reward,
            )

    def render_state(
        self,
        env_state: MathyEnvState,
        action_name: str,
        token_index: int = -1,
        change: Optional[ExpressionChangeRule] = None,
        change_reward: float = 0.0,
        pretty: bool = False,
    ) -> str:
        """Render the given state to a string suitable for printing to a log"""
        changed_problem = env_state.agent.problem
        if change is not None and change.result is not None:
            root = change.result.get_root()
            assert isinstance(root, MathExpression)
            changed_problem = root.terminal_text

        action_name = f"{action_name.lower()}({token_index})"
        output = """{:<25} | {}""".format(action_name.lower(), changed_problem)

        def get_move_shortname(index: int, move: int) -> str:
            if move == 0:
                return "--"
            if move >= len(self.rules):
                return "xx"
            return self.rules[index].code.lower()

        moves_left = str(env_state.agent.moves_remaining).zfill(2)
        valid_rules = self.get_valid_rules(env_state)
        valid_moves = self.get_valid_moves(env_state)
        num_moves = "{}".format(len(np.nonzero(np.array(valid_moves))[0])).zfill(3)
        move_codes = [get_move_shortname(i, m) for i, m in enumerate(valid_rules)]
        moves = " ".join(move_codes)
        reward = f"{change_reward:.2}"
        reward = f"{reward:<5}"
        if pretty:
            return output
        return f"{num_moves} | {moves} | {moves_left} | {reward} | {output}"

    def random_action(
        self,
        expression: MathExpression,
        rule: Optional[Type[BaseRule]] = None,
    ) -> Tuple[int, int]:
        """Get a random action index that represents a particular rule"""

        if rule is not None:
            found = -1
            for rule_idx, r in enumerate(self.rules):
                if isinstance(r, rule):  # type:ignore
                    found = rule_idx
                    break
            if found == -1:
                raise ValueError(
                    "The action {rule} does not exist in the environment rule list"
                )
            all_actions = self.get_actions_for_node(expression, [rule])
            valid_actions = np.nonzero(all_actions[found])[0].tolist()  # type:ignore
            action: int = random.choice(valid_actions)  # type:ignore
            return (found, int(action))

        all_actions = self.get_actions_for_node(expression)
        valid_rules = [i for i, r in enumerate(all_actions) if 1 in r]
        if len(valid_rules) == 0:
            raise ValueError(f"no valid actions for expression: {expression}")
        chosen_rule = random.choice(valid_rules)
        valid_actions = [i for i, r in enumerate(all_actions[chosen_rule]) if r == 1]
        action = random.choice(valid_actions)
        return chosen_rule, action

    def get_initial_state(
        self, params: Optional[MathyEnvProblemArgs] = None, print_problem: bool = True
    ) -> Tuple[MathyEnvState, MathyEnvProblem]:
        """Generate an initial MathyEnvState for an episode"""
        config = params if params is not None else MathyEnvProblemArgs()
        prob: MathyEnvProblem = self.problem_fn(config)
        self.valid_actions_mask_cache = dict()
        self.valid_rules_cache = dict()
        self.parser.clear_cache()
        self.max_moves = self.max_moves_fn(prob, config)

        # Build and return the initial state
        env_state = MathyEnvState(
            problem=prob.text,
            problem_type=self.get_env_namespace(),
            max_moves=self.max_moves,
            num_rules=len(self.rules),
        )
        if print_problem and self.verbose:
            self.print_state(env_state, "initial-state")
        return env_state, prob

    def get_agent_actions_count(self, env_state: MathyEnvState) -> int:
        """Return number of all possible actions"""
        node_count = len(self.parser.parse(env_state.agent.problem).to_list())
        return self.action_size * node_count

    def get_token_at_index(
        self, expression: MathExpression, index: int
    ) -> Optional[MathExpression]:
        """Get the token that is `index` from the left of the expression"""
        count = 0
        result: Optional[MathExpression] = None

        def visit_fn(
            node: BinaryTreeNode, depth: int, data: Any
        ) -> Optional[VisitStop]:
            nonlocal result, count
            result = node  # type:ignore
            if count == index:
                return STOP
            count = count + 1
            return None

        expression.visit_inorder(visit_fn)
        return result

    def get_valid_moves(self, env_state: MathyEnvState) -> List[List[int]]:
        """Get a 2d list describing the valid moves for the current state.

        The first dimension contains the list of known rules in the order that
        they're registered, and the second dimension contains a list of the max
        sequence length size that is 1/0 representing that the node at that index
        for the given rule is valid.
        """
        agent = env_state.agent
        expression: Optional[MathExpression] = None
        try:
            expression = self.parser.parse(agent.problem)
        except InvalidSyntax as err:
            raise_with_history(
                self.get_env_namespace(), err.message, env_state.agent.history
            )
        assert expression is not None
        return self.get_actions_for_node(expression)

    def get_valid_rules(self, env_state: MathyEnvState) -> List[int]:
        """Get a vector the length of the number of valid rules that is
        filled with 0/1 based on whether the rule has any nodes in the
        expression that it can be applied to.

        !!! note

            If you want to get a list of which nodes each rule can be
            applied to, prefer to use the `get_valid_moves` method.
        """
        key = self.to_hash_key(env_state)
        if key in self.valid_rules_cache:
            return self.valid_rules_cache[key]
        expression = self.parser.parse(env_state.agent.problem)
        actions = [0] * len(self.rules)
        for rule_index, rule in enumerate(self.rules):
            nodes = rule.find_nodes(expression)
            actions[rule_index] = 0 if len(nodes) == 0 else 1
        self.valid_rules_cache[key] = actions[:]
        return actions

    def get_rule_from_timestep(self, time_step: MathyEnvStateStep) -> BaseRule:
        return self.rules[time_step.action[0]]

    def get_actions_for_node(
        self,
        expression: MathExpression,
        rule_list: Optional[List[Type[BaseRule]]] = None,
    ) -> List[List[int]]:
        """Return a valid actions mask for the given expression and rule list.

        Action masks are 2d lists of length (num_rules, max_seq_len) where a 0 indicates
        the action is not valid in the current state, and a 1 indicates that it is
        a valid action to take."""
        key = str(expression)
        if rule_list is None and key in self.valid_actions_mask_cache:
            return self.valid_actions_mask_cache[key][:]
        node_count = len(expression.to_list())
        rule_count = len(self.rules)
        actions = [[0] * node_count for _ in range(rule_count)]
        for rule_index, rule in enumerate(self.rules):
            if rule_list is not None:
                if not isinstance(rule, tuple(rule_list)):  # type:ignore
                    continue
            nodes = rule.find_nodes(expression)
            for node in nodes:
                assert node.r_index is not None
                actions[rule_index][node.r_index] = 1
        if rule_list is None:
            self.valid_actions_mask_cache[key] = actions[:]
        return actions

    def to_hash_key(self, env_state: MathyEnvState) -> str:
        """Convert env_state to a string for MCTS cache"""
        return env_state.agent.problem

    def to_action(self, action: Union[int, np.int64, ActionType]) -> ActionType:
        """Resolve a given action input to a tuple of (rule_index, node_index).

        When given an int, it is treated as an index into the flattened 2d action
        space. When given a tuple, it is assumed to be (rule, node)"""
        if isinstance(action, (tuple, list)):
            return action
        token_index = action % self.max_seq_len
        action_index = int((action - token_index) / self.max_seq_len)
        return action_index, int(token_index)

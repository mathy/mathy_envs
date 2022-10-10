## [0.11.4](https://github.com/mathy/mathy_envs/compare/v0.11.3...v0.11.4) (2022-10-10)


### Bug Fixes

* **python:** use typing_extensions for py < 3.8 ([f963652](https://github.com/mathy/mathy_envs/commit/f9636522f71822ef17204988ad31ecf53cbba7bf))

## [0.11.3](https://github.com/mathy/mathy_envs/compare/v0.11.2...v0.11.3) (2022-10-10)


### Features

* **env:** add RestateSubtractionRule to defaults ([c634a0e](https://github.com/mathy/mathy_envs/commit/c634a0efd8c1977f43f65866d3066559106cfdc3))

## [0.11.2](https://github.com/mathy/mathy_envs/compare/v0.11.1...v0.11.2) (2022-10-10)


### Bug Fixes

* **gym:** warning about legacy ragged to array conversion ([60ce708](https://github.com/mathy/mathy_envs/commit/60ce708d37b0f81e45cb0578fe8ab10fe86833c0))

## [0.11.1](https://github.com/mathy/mathy_envs/compare/v0.11.0...v0.11.1) (2022-09-24)


### Bug Fixes

* **gym:** pin version to < 0.26.0 ([ba8137c](https://github.com/mathy/mathy_envs/commit/ba8137cf82f4310d8a57f221692960e1b8118546))


### Features

* **mathy_core:** update to mathy_core >= 0.8.6 ([ede3d8d](https://github.com/mathy/mathy_envs/commit/ede3d8d441116f7d0700261c0b3bb815877aa997))
* **requirements:** remove pydantic in favor of dataclass ([9063747](https://github.com/mathy/mathy_envs/commit/9063747de54f216b080484b84e930f375a3aff96))

# [0.11.0](https://github.com/mathy/mathy_envs/compare/v0.10.0...v0.11.0) (2021-08-11)


### Bug Fixes

* **env:** using "penalize" or "terminal" invalid action responses could loop infiitely ([f7bf88b](https://github.com/mathy/mathy_envs/commit/f7bf88bd42f6b58c0b76c34c240602aa5a57257e))


### chore

* drop MathyWindowObservation and observations_to_window ([75efcac](https://github.com/mathy/mathy_envs/commit/75efcac734051740b28825f166f4232e12b89cec))
* **gym:** drop support for Goal based envs ([f84033e](https://github.com/mathy/mathy_envs/commit/f84033e4159957f6600b9c7a77590d8b694e59e9))


### Features

* **gym:** environments return np.ndarray observations only ([e441dfc](https://github.com/mathy/mathy_envs/commit/e441dfccfea7011551efb3514ea468716e097255))


### BREAKING CHANGES

* drop MathyWindowObservation and observations_to_window helpers for working with windowed observations. They were only used in the legacy custom mathy agents.
* **gym:** this remove the "np_observation" flag from MathyGymEnv. All returned obseravtions are np.ndarrays now.
* **gym:** This removed all Gym Goal-based environments. It's unclear that the implementation I provided worked as expected, and the experimental results were poor.

# [0.10.0](https://github.com/mathy/mathy_envs/compare/v0.9.3...v0.10.0) (2020-11-30)


### Bug Fixes

* **state:** normalize problem type hash in observation ([97b26b5](https://github.com/mathy/mathy_envs/commit/97b26b5df7dc579fb045fcb5e359a7b1812531f6))


### BREAKING CHANGES

* **state:** observation size is 2 floating point values more than before. In order to normalize problem type, we need more than two options or the normalized value will be likely to be 0.0 and 1.0 in some order. By adding two more variations on the input, we get a range of 4 values with at least 2 usually being not 1.0 or 0.0

## [0.9.3](https://github.com/mathy/mathy_envs/compare/v0.9.2...v0.9.3) (2020-11-21)


### Features

* **env:** add preferred_term_commute boolean ([de7de8f](https://github.com/mathy/mathy_envs/commit/de7de8f562949cf145ff86dd63c9e1c9d434874a))
* **env:** add previous_state_penalty boolean ([a23ace1](https://github.com/mathy/mathy_envs/commit/a23ace1d72bfa2c632bb685656cea4256039c4ef))
* **mathy_core:** update range to >= 0.8.2 ([3afb429](https://github.com/mathy/mathy_envs/commit/3afb4299b714cff9ae26debf89d133058744a2ee))

## [0.9.2](https://github.com/mathy/mathy_envs/compare/v0.9.1...v0.9.2) (2020-10-18)


### Bug Fixes

* **MathyEnv:** random_action did not return an action tuple ([09e2db5](https://github.com/mathy/mathy_envs/commit/09e2db58a52f9ee84ff9abbd10bf6cbae11412c5))

## [0.9.1](https://github.com/mathy/mathy_envs/compare/v0.9.0...v0.9.1) (2020-10-18)


### Features

* **MathyGymEnv:** add mask_as_probabilities flag ([07fffd6](https://github.com/mathy/mathy_envs/commit/07fffd6bc810c6ad6cb7adacd2a97246e9e059cc))

# [0.9.0](https://github.com/mathy/mathy_envs/compare/v0.8.4...v0.9.0) (2020-10-11)


### Features

* **env:** add invalid_action_response options ([f120261](https://github.com/mathy/mathy_envs/commit/f120261c8a1f57584085f3b3dacafa2903678e17))
* **gym:** add gym.GoalEnv variants for all mathy envs ([4841b37](https://github.com/mathy/mathy_envs/commit/4841b37abba060539ffb63d134b483054980ae49))

## [0.8.4](https://github.com/mathy/mathy_envs/compare/v0.8.3...v0.8.4) (2020-09-13)


### Features

* **state:** add normalize param to_observation ([4045686](https://github.com/mathy/mathy_envs/commit/4045686fa8af571fc6b8869fb23cfa52a258f9c4))

## [0.8.3](https://github.com/mathy/mathy_envs/compare/v0.8.2...v0.8.3) (2020-09-13)


### Features

* **mathy_core:** update to 0.8.0 with full typings ([3c2e5ad](https://github.com/mathy/mathy_envs/commit/3c2e5adec624407d766ee2a3a409491cac51f98d))

## [0.8.2](https://github.com/mathy/mathy_envs/compare/v0.8.1...v0.8.2) (2020-09-06)


### Features

* **types:** resolve all mypy/pyright/flake8 errors ([6571ac5](https://github.com/mathy/mathy_envs/commit/6571ac5c5c6279beb1c2da27bd92607355502b25))

## [0.8.1](https://github.com/mathy/mathy_envs/compare/v0.8.0...v0.8.1) (2020-09-06)


### Bug Fixes

* **ci:** test suite passes ([83550c7](https://github.com/mathy/mathy_envs/commit/83550c7c3c660758ccaa01357bc362550a19ad1c))

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

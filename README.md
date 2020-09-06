# mathy_envs: Learning environments for solving math problems

[![Build status](https://travis-ci.com/mathy/mathy_envs.svg?branch=master)](https://travis-ci.com/mathy/mathy_envs)
[![codecov](https://codecov.io/gh/mathy/mathy_envs/branch/master/graph/badge.svg)](https://codecov.io/gh/mathy/mathy_envs)
[![Pypi version](https://badgen.net/pypi/v/mathy-envs)](https://pypi.org/project/mathy-envs/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Mathy environments present users/agents with a prompt problem, and are asked to simplify it according to some problem-specific criteria using combinations of simple actions based on the Properties of Numbers.

## 🚀 Quickstart

You can install `mathy_envs` from pip:

```bash
pip install mathy_envs
```


## Semantic Versioning

Mathy Envs tries to be predictable when it comes to breaking changes, so the project uses semantic versioning to help users avoid breakage.

Specifically, new releases increase the `patch` semver component for new features and fixes, and the `minor` component when there are breaking changes. If you don't know much about semver strings, they're usually formatted `{major}.{minor}.{patch}` so increasing the `patch` component means incrementing the last number.

Consider a few examples:

| From Version | To Version | Changes are Breaking |
| :----------: | :--------: | :------------------: |
|    0.2.0     |   0.2.1    |          No          |
|    0.3.2     |   0.3.6    |          No          |
|    0.3.1     |   0.3.17   |          No          |
|    0.2.2     |   0.3.0    |         Yes          |

If you are concerned about breaking changes, you can pin the version in your requirements so that it does not go beyond the current semver `minor` component, for example if the current version was `0.1.37`:

```
mathy_envs>=0.1.37,<0.2.0
```

## Contributors

Mathy Envs wouldn't be possible without the wonderful contributions of the following people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a target="_blank" href="https://www.justindujardin.com/"><img src="https://avatars0.githubusercontent.com/u/101493?v=4" width="100px;" alt=""/><br /><sub><b>Justin DuJardin</b></sub></a></td>
    <td align="center"><a target="_blank" href="https://twitter.com/Miau_DB"><img src="https://avatars3.githubusercontent.com/u/7149899?v=4" width="100px;" alt=""/><br /><sub><b>Guillem Duran Ballester</b></sub></a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

(sec_development)=

# Development

Contributions to Dinf are welcome!
Please make pull requests against our
[git repository](https://github.com/RacimoLab/dinf).

## Installation

Compared with a regular installation via `pip install dinf`,
additional dependencies are required during development, as
developers regularly run the test suite, build the documentation, and assess
whether their code changes conform to style guidelines.
For developers, installation is from the git repository directly,
and a virtual environment is highly recommended. The `requirements.txt`
file in the top-level folder points to the developer dependencies.

```sh
# Clone the repository.
git clone https://github.com/RacimoLab/dinf.git
cd dinf
# Create a virtual environment for development.
python -m venv venv
# Activate the environment.
source venv/bin/activate
pip install --upgrade pip
# Install the developer dependencies.
pip install -r requirements.txt
```

```{note}
Non-developer requirements are listed in the `install_requires` section
of the ``setup.cfg`` file in the top-level folder of the sources.
```

## Continuous integration (CI)

After a pull request is submitted, an automated process known as
*continuous integration* (CI) will:

 * assess if the proposed changes conform to style guidelines (known as *lint* checks),
 * run the test suite,
 * and build the documentation.

The CI process uses
[GitHub Actions](https://docs.github.com/en/free-pro-team@latest/actions)
and the configuration files detailing how these are run can be found under the
`.github/workflows/` folder of the sources.

## Lint checks

The following tools are run during the linting process:

 * [black](https://black.readthedocs.io/), a code formatter
   (code is only checked during CI, not reformatted),
 * [flake8](https://flake8.pycqa.org/),
   a [PEP8](https://www.python.org/dev/peps/pep-0008/) code-style checker,
 * [mypy](http://mypy-lang.org/), a static type checker.

Each of these tools can also be run manually from the top-level folder of the
sources. The `setup.cfg` file includes some project-specific configuration
for each of these tools, so running them from the command line should match
the behaviour of the CI checks.

## Test suite

A suite of tests is included in the `tests/` folder.
The CI process uses the `pytest` tool to run the tests, which can also be run
manually from the top-level folder of the sources.

```sh
python -m pytest -v tests --cov=dinf --cov-report=term-missing
```

## Building the documentation

The Dinf documentation is built with [jupyter-book](https://jupyter-book.org/),
which uses [sphinx](https://www.sphinx-doc.org/).
Much of the documentation is under the `docs/` folder, written in the
[MyST](https://myst-parser.readthedocs.io/en/latest/) flavour of Markdown,
and is configured in the `docs/_config.yml` file.
In contrast, the API documentation is automatically generated from "docstrings"
in the Python code that use the
[reStructuredText](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html)
format.
Finally, some documentation files are ipython notebooks that get executed and
converted into markdown by jupyter-book. A couple of these notebooks take a
long time to run, so they are excluded from execution in the
jupyter-book configuration file and run manually as required.

To build the documentation locally, run the following from the top-level folder.

```sh
make -C docs
```


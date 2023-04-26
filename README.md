# End-to-end Project Template (On-prem - Polyaxon)

__Customised for `BIPO`__.

__Project Description:__ Demand Forecasting

This template that is also accompanied with an end-to-end guide was
generated and customised using the
following
[`cookiecutter`](https://cookiecutter.readthedocs.io/en/stable/)
template:
https://github.com/aisingapore/ml-project-cookiecutter-onprem-poly

The contents of the guide have been customised
according to the inputs provided upon generation of this repository
through the usage of [`cruft`](https://cruft.github.io/cruft/),
following instructions detailed
[here](https://github.com/aisingapore/ml-project-cookiecutter-onprem-poly/blob/master/README.md)
.

Inputs provided to `cookiecutter`/`cruft` for the generation of this
template:

- __`project_name`:__ BIPO
- __`description`:__ Demand Forecasting
- __`repo_name`:__ bipo
- __`src_package_name`:__ bipo
- __`src_package_name_short`:__ bipo
- __`harbor_registry_project_path`:__ registry.aisingapore.net/100e-bipo
- __`author_name`:__ bipo
- __`open_source_license`:__ No license file

## End-to-end Guide

This repository contains a myriad of boilerplate codes and configuration
files. On how to make use of these boilerplates, this repository
has an end-to-end guide on that.
The guide's contents are written in Markdown formatted files, located
within `aisg-context/guide-site` and its subdirectories. While the
Markdown files can be viewed directly through text editors or IDEs,
the contents are optimised for viewing through
[`mkdocs`](https://www.mkdocs.org) (or
[`mkdocs-material`](https://squidfunk.github.io/mkdocs-material)
specifically)
.
A demo of the site for the guide can be viewed
[here](https://aisingapore.github.io/ml-project-cookiecutter-onprem-poly)
.

To spin up the site on your local machine, you can create a virtual
environment to install the dependencies first:

```bash
$ conda create -n aisg-eptg-onprem-poly-guide python=3.8.13
$ conda activate aisg-eptg-onprem-poly-guide
$ pip install -r aisg-context/guide-site/mkdocs-requirements.txt
```

After creating the virtual environment and installing the required
dependencies, serve it like so:

```bash
$ mkdocs serve --config-file aisg-context/guide-site/mkdocs.yml
```

The site for the guide will then be viewable on
[`http://localhost:8000`](http://localhost:8000).

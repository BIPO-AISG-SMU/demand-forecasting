# Preface

## Repository Setup

This repository provides an end-to-end template for AI
Singapore's AI engineers to onboard their AI projects.
Instructions for generating this template is detailed in the
`cookiecutter` template's repository's
[`README.md`](https://github.com/aisingapore/ml-project-cookiecutter-onprem-poly/blob/master/README.md).

While this repository provides users with a set of boilerplates,
here you are also presented with a linear guide on
how to use them. The boilerplates are rendered and customised
when you generated this
repository using [`cruft`](https://cruft.github.io/cruft/).

!!! info
    You can begin by following along the guide as it brings you through
    a simple problem statement and
    once you've grasp what this template has to offer,
    you can deviate from it as much as you wish
    and customise it to your needs.

Since we will be making use of this repository in multiple
environments, __ensure that this repository is pushed to a
remote__.
Most probably you will be resorting to
[AI Singapore's GitLab instance](https://gitlab.aisingapore.net/) as
the remote.
Refer to
[here](https://docs.gitlab.com/ee/user/project/working_with_projects.html#create-a-project)
on creating a blank remote repository (or project in GitLab's term).
After creating the remote repository, retrieve the remote URL and push
the local repository to remote:

```bash
$ git init
$ git remote add origin <REMOTE_URL>
$ git add .
$ git config user.email "<YOUR_AISG_EMAIL>"
$ git config user.name "<YOUR_NAME>"
$ git commit -m "Initial commit."
$ git push -u origin master
```

## Guide's Problem Statement

For this guide, we will work towards building a predictive model that is
able to conduct sentiment classification for movie reviews.
The model is then to be deployed through a REST API and used for batch
inferencing as well.
The raw dataset to be used is obtainable through a S3 bucket;
instructions for downloading the data into your development environment
are detailed under
["Data Storage & Versioning"](./06-data-storage-versioning.md),
to be referred to later on.

## On-Premise Resources

In AI Singapore, we have the flexibility and capability to leverage on
cloud resources (like GCP or Azure) and/or on-premise resources
(located in our own data centre). While the usage of cloud platform
services are usually collated within a single dashboard or project,
the on-premise services/resources are presented in a more isolated
manner, with different entry points. In the following sections, we will
cover the different entry points and accounts needed to access the
differing resources/services/components.

### Azure (for LDAP)

Upon onboarding, engineers under AI Singapore's Industry Innovation and
Products pillar would have received credentials for their own Azure
account. This account is used for LDAP authentication, providing users
with access to the following services:

- [Microsoft Azure Portal](http://portal.azure.com)
- [AI Singapore's GitLab instance](https://gitlab.aisingapore.net)
- [AI Singapore's Rancher dashboard](https://rancher.aisingapore.net)
- [AI Singapore's Harbor registry](https://registry.aisingapore.net)

### Miscellaneous Credentials

- Polyaxon dashboard (exclusive to each project)
- MLflow Tracking server (exclusive to each project)
- ECS (S3) bucket (exclusive to each project)

!!! info
    Projects are managed and
    provisioned by AI Singapore's Platforms team.
    Should you require access to any of the aforementioned services and
    you do not have the respective accounts, please
    contact `mlops@aisingapore.org`.

In order to interact with the on-premise resources, we will need to
configure the components for it. Unlike cloud services, this template
interacts with each component separately to allow for flexibility.

### Container Registry

Using the Docker Engine CLI, one may pull or push images to container
registries. However, to pull from or push to any registry other than
[Docker Hub](https://hub.docker.com), we would need to authenticate
first. In this case, one would need to authenticate the Docker CLI with
AI Singapore's Harbor registry using the relevant Azure credentials:

```bash
$ docker login registry.aisingapore.net
```

### AWS CLI

We will be using AWS CLI to access AI Singapore's on-premise
object storage system:
[ECS](https://www.dell.com/en-sg/dt/storage/ecs/index.htm#tab0=0&tab1=0).
ECS uses the S3 protocol hence it is possible to utilise AWS' CLI for
S3 operations.

The AWS CLI offers multiple avenues to pass credentials for the CLI,
ranging from setting environmental variables to populating specific
files within the `~/.aws`  directory.

For your personal machine, it will be more robust and secure to generate the
credential file on your machine with the inbuilt `configure` function.

```bash
$ aws configure
```

!!! note
	Within AI Singapore's context, leave the `Default region name` and
    `Default output format` parameters empty.

While we have configured these credentials, access to ECS
using the AWS CLI requires an additional flag: `--endpoint-url`. Say
you would like to list resources under your project's bucket, you would
have to specify the command like so:

```bash
$ aws s3 ls s3://my-project-bucket --endpoint-url="https://necs.nus.edu.sg"
```

__Reference(s):__

- [Okta - What is LDAP & how does it work?](https://www.okta.com/sg/identity-101/what-is-ldap)
- [AWS Docs - Using the AWS CLI examples](https://docs.aws.amazon.com/cli/latest/userguide/welcome-examples.html)

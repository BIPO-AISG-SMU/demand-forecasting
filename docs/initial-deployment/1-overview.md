# Project Overview

## Purpose of Documentation

List scope of documentation - what it does and does not include.

## Solution Architecture

Insert ML pipeline diagram here.

## Project File Structure

The tables below summarizes all the notable locations in the project code base:

### Data

| Path | Description |
| :- | - |
| **Data Folders**<br>**(data/\*)** | All data used in the project should and will be in this folder. |
| 01_raw | Raw project data should be extracted into this folder. If multiple projects exists, put them into further subfolders. |

### Configuration

| Path | Description |
| :- | - |
| **Project Configuration**<br>**(conf/\*)** | All .yml configurations will be in this folder. |
| **conf/base/*** | Default Kedro [configuration environment](https://kedro.readthedocs.io/en/stable/kedro_project_setup/configuration.html#local-and-base-configuration-environments). Also contains configurations for non-Kedro pipelines. |

### Source Code

| Path | Description |
| :- | - |
| **Source Code**<br>**(src/\*)** | Actual source code of the project are in this folder. |
| requirements.txt | Project dependencies that can be used by [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) for installation.<br><br>**Note:** This complements the 100e-bipo-conda-env.yml file in the root folder used by Conda. Packages that are distributed through Conda channels should preferably be installed via Conda, and only those that do not will be installed via pip. This is so as to rely on Conda to solve environment dependencies for compatible versions. |
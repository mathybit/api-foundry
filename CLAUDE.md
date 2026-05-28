# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.



# Repository Purpose

This is a repository for building a Python / Flask API framework that will be used to deploy various services (compute services or PyTorch-based machine learning models).

The API code itself must be built in such a way that models can be easily integrated without any substantial changes to the API code.

## Folder structure

- `api` - holds the API-specific code
- `config` - holds various API configuration options
- `doc` - holds documentation for the API and individual models or services that are deployed within it
- `neural` - this folder will contain reusable pieces of code that are shared among the deep learning models (e.g. custom layers, other neural utilities needed for inference)
- `services` - this folder will hold the deployed services / models. Each model will sit in its own subdirectory, e.g. services/model_name, and will also have its own configuration file.
- `utils` - this folder will create various shared utility functions that fall outside "neural-specific" utils / custom layers
- `scripts` - this folder will contain various scripts
- `tests` - this folder will contain test scripts

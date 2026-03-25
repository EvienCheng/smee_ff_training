# smee_ff_training

Training notebooks for fitting custom force field functional forms using **SMEE** and **Descent** with a custom SMIRNOFF plugin.

This repository contains training workflows for four custom valence functional forms:

- Lee–Krimm improper potential  
- Harmonic angle potential  
- Harmonic height improper potential  
- Two-minima improper potential  

These functional forms are implemented through a custom **SMIRNOFF plugin** and trained using **SMEE** force field optimization tools.

---

## Overview

This repository provides Jupyter notebooks and utilities for training force field parameters using:

- SMEE
- Descent
- OpenFF Toolkit
- OpenFF Interchange
- Custom SMIRNOFF plugins

The goal is to evaluate alternative functional forms for valence terms and compare their performance during force field fitting.

---

## Installation

Create the conda environment using the provided YAML file:

```bash
conda env create -f smee_training.yml
conda activate smee_training
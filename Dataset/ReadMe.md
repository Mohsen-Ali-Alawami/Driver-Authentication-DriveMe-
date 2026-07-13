# DriveMe Dataset

## Overview

This directory contains the complete dataset used in the paper:

**DriveMe: Towards Lightweight and Practical Driver Authentication System Using Single-Sensor Pressure Data**

The dataset consists of pressure measurements collected from **12 users** using a **30-sensor pressure matrix** under different seating conditions and recording durations.

---

# Dataset Structure

The dataset is organized into two main categories according to the sensor placement:

```
Dataset/
│
├── Belt/
│   ├── Size A/
│   │   ├── U1/
│   │   ├── U2/
│   │   ├── ...
│   │   └── U12/
│   │
│   ├── Size B/
│   ├── Size C/
│   └── Size D/
│
└── Seat/
    ├── Size A/
    ├── Size B/
    ├── Size C/
    └── Size D/
```

Each user folder contains the recorded pressure samples collected during multiple experiments.

---

# Dataset Description

The dataset was collected from **12 different drivers**.

Each pressure sample consists of measurements from

- **30 pressure sensors (30 columns)**

The recording was repeated

- **10 times for each user**

For every experimental configuration.

---

# Experimental Configurations

## Belt-based Dataset

| Size | Recording Time | Samples per Experiment | Total Samples |
|------|----------------|-------------------------|---------------|
| Size A | 5 s | 50 | 6000 |
| Size B | 3 s | 30 | 3600 |
| Size C | 2 s | 20 | 2400 |
| Size D | 1 s | 10 | 1200 |

---

## Seat-based Dataset

| Size | Recording Time | Samples per Experiment | Total Samples |
|------|----------------|-------------------------|---------------|
| Size A | 10 s | 100 | 12,000 |
| Size B | 5 s | 50 | 6000 |
| Size C | 3 s | 30 | 3600 |
| Size D | 1 s | 10 | 1200 |

---

# Data Format

Each sample contains

- Pressure values from **30 sensors**
- One row corresponds to one timestamp (sample)
- Columns correspond to individual pressure sensors

Example:

```
S1,S2,S3,...,S30
103,98,105,...,115
104,97,106,...,114
...
```

---

# Naming Convention

```
Belt/
    Size A/
        U1/
        U2/
        ...
        U12/

Seat/
    Size A/
        U1/
        ...
        U12/
```

where

- **Belt** : Belt-mounted pressure sensor
- **Seat** : Seat-mounted pressure sensor
- **Size A** : Longest recording duration
- **Size B**
- **Size C**
- **Size D** : Shortest recording duration
- **U1–U12** : User IDs

---

# Dataset Statistics

| Property | Value |
|----------|------:|
| Number of users | 12 |
| Number of sensors | 30 |
| Experiments per user | 10 |
| Sensor locations | Belt, Seat |
| Recording durations | 1 s – 10 s |
| Total experimental configurations | 8 |

---

# Citation

If you use this dataset, please cite our paper.

```bibtex
@article{ALAWAMI20252361,
title = {DriveMe: Towards Lightweight and Practical Driver Authentication System Using Single-Sensor Pressure Data},
journal = {CMES - Computer Modeling in Engineering and Sciences},
volume = {143},
number = {2},
pages = {2361-2389},
year = {2025},
issn = {1526-1492},
doi = {https://doi.org/10.32604/cmes.2025.063819},
url = {https://www.sciencedirect.com/science/article/pii/S1526149225001213},
author = {Mohsen Ali Alawami and Dahyun Jung and Yewon Park and Yoonseo Ku and Gyeonghwan Choi and Ki-Woong Park}
}
```

---

# License

This dataset is released for **research and educational purposes only**.

Please cite the corresponding paper when using this dataset.

---

# Contact

Prof. Mohsen Ali Alawami

Division of Computer Engineering

Hankuk University of Foreign Studies (HUFS)

Republic of Korea

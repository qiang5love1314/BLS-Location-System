# BLS-Location-System
## Overview  
This repository contains the implementation of the **BLS-Location algorithm** described in our paper: *"BLS-Location: A Wireless Fingerprint Localization Algorithm Based on Broad Learning"*. BLS-Location leverages **broad learning systems (BLS)** and **channel state information (CSI)** to address challenges in wireless fingerprint localization, such as data loss, noise interference, and high training complexity.  

## Abstract  
The rapid growth of location-based services in indoor environments has driven interest in wireless fingerprint localization due to its high precision and ease of implementation. However, current methods face significant challenges, including:  
- **Data loss and noise interference** in the fingerprint database.  
- **Time-consuming offline training phases**.  

BLS-Location addresses these challenges by combining advanced data processing techniques and a broad learning system to achieve efficient and accurate indoor localization.  

Key contributions include:  
- **Offline Training Phase**:  
  - Using **Kalman filter** and **expectation-maximization (EM) algorithm** to complete and denoise fingerprint data.  
  - Applying **principal component analysis (PCA)** to reduce CSI data complexity.  
  - Training weights with a **broad learning system (BLS)** for fast and efficient processing.  
- **Online Localization Phase**:  
  - A novel probabilistic regression method based on BLS results to estimate locations.  

Experimental results demonstrate that BLS-Location significantly reduces training time while maintaining high accuracy compared to traditional machine learning algorithms and state-of-the-art methods.  

## Usage  
To use this repository, follow these steps:  
- `LabMain.py`: Implements the proposed algorithm for the Lab Scenario.  
- `MeetingMain.py`: This is for the Meeting Room Scenario.
- You can find the datasets on my homepage, and see the repositories of "CSI-dataset".

If you use this work for your research, you may want to cite
```
@article{zhu2021bls,
  title={{BLS-Location: A Wireless Fingerprint Localization Algorithm Based on Broad Learning}},
  author={Zhu, Xiaoqiang and Qiu, Tie and Qu, Wenyu and Zhou, Xiaobo and Atiquzzaman, Mohammed and Wu, Dapeng},
  journal={IEEE Transactions on Mobile Computing},
  volume={22},
  number={1},
  pages={115--128},
  year={2023}
}
```

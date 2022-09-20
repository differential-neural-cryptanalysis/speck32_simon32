# Enhancing Differential-Neural Cryptanalysis

This repository provides the supplementary code and data to the paper entitled "Enhancing Differential-Neural Cryptanalysis" [\[2\]](https://eprint.iacr.org/2021/719).

Concretely, it includes the following.

* the implementation of the differential-neural key-recovery attacks on the round-reduced (11, 12, 13 rounds) block cipher Speck32/64 (in `speck/key_recovery.py`).

* the implementation of the differential-neural key-recovery attacks on the round-reduced (16 rounds) block cipher Simon32/64 (in `simon/key_recovery.py`).

* pre-trained neural distinguishers on round-reduced Simon32/64

  - `simonNDvsDD/SENet`: NDs employing Squeeze-and-Excitation blocks (SENet) and based on ResNeXt backbone (SE-ResNeXt)

  - `simonNDvsDD/DenseNet`: NDs using Dense Network (DenseNet)

  - `simonNDvsDD/ResNet_lr_1e-3_1e-5`: NDs using Residual Networks (ResNet), trained with a learning rate scheduler cyclic\_lr(10,0.001,0.00001)

  - `simonNDvsDD/ResNet_lr_2e-3_1e-4`: NDs using Residual Networks (ResNet), trained with a learning rate scheduler cyclic\_lr(10,0.002,0.0001)

* the source code to generate the full DDTs given an input difference (in `simonNDvsDD/simon_ddt.cpp`)

* the comparison between ND and DD on Simon32/64 using a key ranking procedure (in `simonNDvsDD/key_rank.py` and `simonNDvsDD/key_rank_ddt.py`)

* the deeper look into ND and DD on round-reduced Simon32/64, including 
  the finding of a linear combiner, the key-average distinguisher based on (n - 2)-round DDT,
  and the ploting of the scores given by various distinguishers (in `simonNDvsDD/combine_AD_DD_VD_VV_parallel.py`)

## Tested configuration
- tensorflow == 2.4.1
- keras == 2.4.3
- h5py == 2.10.0
- numpy == 1.19.2
- seaborn == 0.11.0
- pandas == 1.1.3

## References

Many of the codes are build upon the codes in the GitHub repository named [deep_speck](https://github.com/agohr/deep_speck) that provides Supplementary code and data to the paper [\[1\]](https://eprint.iacr.org/2019/037).

Please refer to the academic paper [\[2\]](https://eprint.iacr.org/2021/719) reporting the technical details and experimental results in this repository.

[1] Aron Gohr: Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning. CRYPTO (2) 2019: 150-179 https://eprint.iacr.org/2019/037

[2] Zhenzhen Bao, Jian Guo, Meicheng Liu, Li Ma, Yi Tu: Enhancing Differential-Neural Cryptanalysis. IACR Cryptol. ePrint Arch. 2021: 719 (2021) https://eprint.iacr.org/2021/719

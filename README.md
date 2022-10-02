# Simulation-Guided Beam Search for Neural Combinatorial Optimization

This repository is the official implementation of **Simulation-Guided Beam Search for Neural Combinatorial Optimization** (NeurIPS 2022). <br>
> https://arxiv.org/abs/2207.06190



![sgbs_3steps](https://user-images.githubusercontent.com/104659627/182066947-70225ccd-0ec9-4188-8ada-abfccafcc751.png)



<br>

## Inference with SGBS/SGBS+EAS method

### 📋 TSP

#### SGBS
```
cd ./TSP/2_SGBS  
python3 test.py
```

#### SGBS+EAS
```
cd ./TSP/3_SGBS+EAS  
python3 test.py
```

### 📋 CVRP

#### SGBS
```
cd ./CVRP/2_SGBS  
python3 test.py
```

#### SGBS+EAS
```
cd ./CVRP/3_SGBS+EAS  
python3 test.py
```

### 📋 FFSP

To run SGBS and SGBS+EAS for FFSP, you have to download and unpack FFSP.tar.gz first.  
See *Requirements - FFSP trained model & dataset* section.  

#### SGBS
```
cd ./FFSP/2_SGBS  
python3 test_ffsp20.py
python3 test_ffsp50.py
python3 test_ffsp100.py
```

#### SGBS+EAS
```
cd ./FFSP/3_SGBS+EAS  
python3 test_ffsp20.py
python3 test_ffsp50.py
python3 test_ffsp100.py
```

<br>

## Inference with Greedy, Sampling, Beam Search, MCTS or SGBS method for CVRP

```
cd ./CVRP/2_SGBS  
python3 test.py -disable_aug --mode greedy
python3 test.py -disable_aug --mode sampling
python3 test.py -disable_aug --mode obs
python3 test.py -disable_aug --mode mcts
python3 test.py -disable_aug --mode sgbs
```

> -disable_aug: disables instance augmentation  
> --mode: specifies inference method, (obs means original beam search method)   
  
If you want test small number of test episodes, you can use --ep option. For example, if you want test just 10 episodes with greedy method,
> python3 test.py -disable_aug --mode greedy --ep 10  
  
Note: MCTS takes a lot of time compared to other methods.


<br>

## Requirements - FFSP trained model & dataset
Download FFSP trained model and dataset file from https://drive.google.com/file/d/1TdkeErG1FCUMxoe8ENpxiWwUopPIUikb/view?usp=sharing.  
Move FFSP.tar.gz into your root folder and unpack it.  
```
tar -xvzf FFSP.tar.gz
```

<br>

## Language and Libraries
> python 3.8.6  
> torch 1.11.0

<!--
**sgbs-neurips/sgbs-neurips** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

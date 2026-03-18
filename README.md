Thiémokho Fall
# 📧 thiemokho.fall@aims-senegal.org
## 🎓 Master of Science in Mathematical Sciences
## African Institute for Mathematical Sciences (AIMS) — Senegal
------------------------------------------

Supervised by: Dr. Yaé Olatoundji Gaba
AI Research and Innovation Nexus for Africa (AIRINA) · AIRINA Labs by AI.Technipreneurs · Cotonou, Bénin
------------------------------------------------------------
# 📋 Abstract
In the current age of AI, the growing reliance on complex black-box models has created a trade-
off between predictive performance and interpretability. Symbolic regression offers a compelling
alternative by discovering analytical expressions that explain data-driven relationships. This essay
explores the mathematical foundations and algorithmic techniques behind symbolic regression,
surveys its applications in scientific and industrial domains, and compares its strengths and limi-
tations with those of classical machine learning models. Through practical examples and a review
of current tools, we argue that symbolic regression plays a key role in building transparent and
trustworthy AI systems.

🎓 Master of Science in Mathematical Sciences at AIMS Senegal
# Symbolic regression  on HepMC data
🔍 Project overview

This repository presents a comparative study between traditional machine learning methods and symbolic regression for modeling data from the HepMC file.

The main objective is to evaluate the ability of symbolic regression to:produce interpretable mathematical expressions,while maintaining predictive performance comparable to traditional machine learning models.
The gplearn package was used:

- Based on genetic programming

- Use of predefined mathematical functions

- Limited interpretability for complex expressions


📂 Dataset: HepMC

1. Description of the dataset

  The HepMC dataset is a standard format widely used in high-energy physics to represent events.
  Each event contains detailed information on:the particles produced,their kinematic properties (energy, momentum, mass),
  the interaction vertices,as well as derived physical observables.

In this work, the HepMC file was used as a data source to construct a supervised regression problem.

2. Preparetion
   Before conducting any symbolic regression law research, we verified the relativistic consistency of the data by testing the energy-momentum relationship.
  This step ensures that SR operates on a physically valid data set and allows its results to be interpreted as a genuine rediscovery or extension of physical laws.
3. Result:
   When mass and Cartesian components are used as input variables, symbolic regression unambiguously identifies the exact relativistic relationship linking the particle's energy, momentum, and mass.

  #### See SymbolicRegression_FALL_thesis.ipynb
#  Symbolic Regression vs. Classical Machine Learning

 ## Comparative study of interpolation and extrapolation on synthetic and real data


This project aims to analyze the behavior of symbolic regression (SR) in comparison with several classical machine learning models, in contexts of interpolation and extrapolation.

The main objective is to determine:

under what conditions symbolic regression is able to recover an underlying analytical law,

how it behaves on noisy real data for which no simple structure exists.

The models compared are:

- Linear regression

 - Random forests

- Support vector regression (SVR)

- Neural networks

🧪 Datasets
# Synthetic data (controlled experiments)

Three types of analytical functions are generated to evaluate the models' ability to recover the true structure of the data: polynomial function, trigonométric functions,exponential functions

These datasets allow for the evaluation of:

- the quality of interpolation,

- the capacity for extrapolation,

- the explicit recovery of the generating law.


## Real data:

a) Diabetes dataset (scikit-learn):

- Target variable: disease progression indicator

- Explanatory variables: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6.

b) Financial data (AAPL stock – Yahoo Finance):
- Target variaable: y_t = Return_t
- In symbolic regression, only the first explanatory variable is retained:x_0= Return_{t-1}

The problem is thus formaled as follows: Return_t​=f(Return_{t−1}​).
This configuration allows us to test the possible existence of a simple analytical law governing the dynamics of financial returns.

⚙️ Methodology

For each dataset:

The data is separated into training, interpolation testing, and extrapolation testing sets.

The models are trained under comparable conditions.

Performance is evaluated using:mean square error (MSE),coefficient of determination 


For symbolic regression:

- the analytical expressions discovered are extracted,

- their mathematical behavior is analyzed (stability, singularities, extrapolation).


📊 Key findings
## Synthetic data:

Symbolic regression accurately identifies the underlying polynomial and trigonometric laws.It achieves performance comparable to, or even superior to, black-box models in interpolation and extrapolation when the actual structure is simple and regular.

## Real data:

Diabetes: no model achieves high performance, confirming the complex and noisy nature of the problem.

Finance: symbolic regression fits the data very well in interpolation, but often produces mathematically unstable expressions, leading to very poor performance in extrapolation (highly negative R^2 values).

These results show that symbolic regression reveals the absence of a simple analytical law in financial returns, rather than a purely numerical failure.


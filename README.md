# 🧬 Symbolic Regression: A Methodological Framework for Interpretable Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AIMS](https://img.shields.io/badge/AIMS-Senegal-orange.svg)](https://aims-senegal.org/)

> *Research thesis exploring symbolic regression as an interpretable alternative to classical machine learning methods*

---

## 📚 Table of Contents

- [📖 About](#-about)
- [🎯 Objectives](#-objectives)
- [🔬 Methodology](#-methodology)
- [📊 Key Results](#-key-results)
- [🚀 Installation](#-installation)
- [🔍 Applications](#-applications)
- [📈 Comparisons](#-comparisons)
- [📝 Citation](#-citation)
- [👥 Author](#-author)
- [📄 License](#-license)

---

## 📖 About

This repository contains the code and experiments associated with my research thesis on **Symbolic Regression (SR)** conducted at the **African Institute for Mathematical Sciences (AIMS) - Senegal**.

### What is Symbolic Regression?

Symbolic regression is a machine learning technique that, unlike "black box" methods (neural networks, random forests), generates **explicit mathematical expressions** relating input variables to the target.

**Example:**
```python
# Instead of an opaque model, SR produces:
f(x, y) = 2.5*x² + sin(y) - 3.1

# A readable, interpretable, and verifiable formula!
```

### 🌟 Why It Matters

- ✅ **Interpretability**: Explicit formulas understandable by domain experts
- ✅ **Law Discovery**: Reveals hidden physical or biological relationships
- ✅ **Generalization**: Better extrapolation beyond the training domain
- ✅ **Data Efficiency**: Performs well even with few samples
- ✅ **Scientific Validation**: Results can be theoretically verified

---

## 🎯 Objectives

This thesis aims to:

1. **Analyze the mathematical and algorithmic foundations** of symbolic regression
2. **Compare SR with classical ML methods** (SVR, Random Forest, Neural Networks)
3. **Evaluate performance on synthetic and real data** (diabetes, finance, physics)
4. **Demonstrate scientific discovery capability** through rediscovery of physical laws
5. **Provide a complete methodological framework** for practical use of SR

---

## 🔬 Methodology

### Three Algorithmic Paradigms Explored

#### 1️⃣ **Genetic Programming (GP)**
Population of expressions evolving through natural selection, crossover, and mutation.

```python
from gplearn.genetic import SymbolicRegressor

sr = SymbolicRegressor(
    population_size=5000,
    generations=50,
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos'],
    parsimony_coefficient=0.01,
    random_state=42
)
sr.fit(X_train, y_train)
print(sr._program)  # Display discovered formula
```

**Advantages:** Broad exploration, discovery of unexpected structures  
**Limitations:** High computational cost, premature convergence

#### 2️⃣ **Monte Carlo Tree Search (MCTS)**
Sequential construction guided by exploration-exploitation balance (UCB1).

```python
# Node-by-node construction with intelligent exploration
# UCB1(n) = reward_mean(n) + C * sqrt(ln(N) / visits(n))
```

**Advantages:** Focus on promising regions, continuous improvement  
**Limitations:** Memory cost, dependence on rollout quality

#### 3️⃣ **Neuro-Symbolic Method: Equation Learner (EQL)**
Neural network architecture that learns equations while maintaining symbolic structure.

```python
# EQL: Deep learning architecture with custom activation functions
# representing mathematical operators (+, -, ×, ÷, sin, cos, etc.)

from eql import EQL

model = EQL(
    num_layers=4,
    hidden_units=128,
    operator_set=['add', 'mul', 'sin', 'div'],
    learning_rate=0.001
)

# The network weights converge to 0 or 1, revealing the symbolic structure
model.fit(X_train, y_train, epochs=100)

# Extract symbolic expression from trained network
symbolic_expr = model.extract_equation()
print(f"Discovered equation: {symbolic_expr}")
```

**How EQL Works:**

1. **Custom Activation Functions**: Each hidden layer uses activation functions corresponding to mathematical operators (identity for addition/subtraction, multiplication, division, sin, cos, exp, etc.)

2. **Weight Regularization**: L1 regularization pushes weights toward 0 or 1, creating a sparse network that represents a symbolic expression

3. **Equation Extraction**: After training, non-zero connections are interpreted as the discovered mathematical formula

**Advantages:**
- ✅ Gradient-based optimization (faster than GP)
- ✅ Leverages deep learning infrastructure (GPU acceleration)
- ✅ Differentiable approach allows backpropagation
- ✅ Maintains interpretability through sparse structure

**Limitations:**
- ⚠️ Limited to predefined operator set in architecture
- ⚠️ May require careful hyperparameter tuning
- ⚠️ Less flexible than pure symbolic methods for complex expressions

**EQL Architecture Example:**
```
Input Layer (features)
    ↓
Hidden Layer 1: [×, +, sin, div] activations
    ↓
Hidden Layer 2: [×, +, sin, div] activations
    ↓
Hidden Layer 3: [×, +, sin, div] activations
    ↓
Output Layer (prediction)

After training, sparse weights reveal: f(x,y) = sin(x) + 2*y
```

---

## 📊 Key Results

### 🧪 Synthetic Data

| Target Function | SR (gplearn) | SVR | Random Forest | Neural Net |
|-----------------|--------------|-----|---------------|------------|
| `x³ - 2x² + x` | **✅ Exact rediscovery** | ❌ Approximation | ❌ Overfitting | ❌ Black box |
| `sin(2πx)` | **✅ sin(6.28*x)** | ✅ Good approx. | ⚠️ Unstable | ✅ Good approx. |
| `exp(-x²)` | **✅ exp(-x²)** | ⚠️ Edge errors | ✅ Correct | ✅ Correct |

**Conclusion:** SR rediscovers exact functions when operators are available.

---

### 🏥 Real Data: Diabetes (UCI Dataset)

**Problem:** Predict disease progression from 10 clinical variables.

```python
# Expression found by SR:
f_SR = exp(x_0 + 5.03)  # where x_0 = age

# Performance (MSE):
# SR:           5832
# Neural Net:   2900  ✅ Best
# Random Forest: 3100
# SVR:          3500
```

**Analysis:** Simplistic univariate expression. Multivariate and noisy data require more flexible methods.

---

### 💹 Real Data: Finance (Apple Stock Returns)

**Problem:** Predict daily return `y_t` from `y_{t-1}`.

```python
# Expression found by SR:
f_SR = x₀² * sin(-1.15 / x₀)

# Interpolation: ✅ Correct
# Extrapolation: ❌ Unstable (divergence)
```

**Analysis:** No simple and stable analytical relationship between successive returns. SVR shows better robustness.

---

### ⚛️ Scientific Application: Rediscovery of Energy-Momentum Relation

**Context:** Real CERN data (HepMC format), charged particles (pions π±).

**Objective:** Rediscover the relativistic relation `E = f(pₓ, pᵧ, p_z, m)`.

```python
# Expression discovered by SR (gplearn):
E_SR = sqrt(X₀² + X₁² + X₂² + X₃²)

# with:
#   X₀ = pₓ  (momentum x)
#   X₁ = pᵧ  (momentum y)
#   X₂ = p_z (momentum z)
#   X₃ = m   (mass)
```

**🎉 Result:** Matches **exactly** Einstein's formula:

```
E = √(p²c² + m²c⁴)  in natural units (c=1)
```

**Performance:**
- ✅ Final MSE: `4 × 10⁻⁵` (near-perfect)
- ✅ Convergence: 25 generations
- ✅ Tree size: 16 nodes

**💡 Conclusion:** SR rediscovered a fundamental physical law without any prior knowledge of special relativity!

---

## 🚀 Installation

### Prerequisites

```bash
Python >= 3.8
pip >= 21.0
```

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/thiefall/Thiemokho-Code-essay-AIMS-Senegal.git
cd Thiemokho-Code-essay-AIMS-Senegal

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt
```

### Main Packages

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
gplearn>=0.4.2
sympy>=1.9
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
tensorflow>=2.8.0  # For EQL
pysr>=0.11.0  # Optional (Julia backend)
```

---


### 📁 Available Notebooks

| Notebook | Description |
|----------|-------------|
| `Comparaison_SR\Others.ipynb` | Experiments on synthetic functions |
| `SymbolicRegression_Fall_thesis.ipynb` | Rediscovery of E = √(p²+m²) with CERN data |

### 🎯 Quick Example

```python
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split

# Generate synthetic data
X = np.random.uniform(-5, 5, (1000, 2))
y = X[:, 0]**2 + np.sin(X[:, 1])  # Target function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Configure symbolic regressor
sr = SymbolicRegressor(
    population_size=5000,
    generations=50,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    parsimony_coefficient=0.01,
    random_state=42,
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'sin', 'cos'],
    verbose=1
)

# Train
sr.fit(X_train, y_train)

# Display discovered expression
print("Discovered expression:")
print(sr._program)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_pred = sr.predict(X_test)
print(f"\nMSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")

# Convert to SymPy for simplification
from sympy import simplify, sympify
expr_sympy = sympify(str(sr._program))
expr_simplified = simplify(expr_sympy)
print(f"\nSimplified expression:\n{expr_simplified}")
```


## 🔍 Applications

### 🔬 Scientific Discovery

Symbolic regression is particularly suited for:

- **Physics**: Rediscovery of laws (kinematics, thermodynamics, relativity)
- **Biology**: Population dynamics modeling, enzyme kinetics
- **Chemistry**: Structure-activity relationships, equations of state
- **Engineering**: Empirical laws, material models

### 📊 Use Cases

| Domain | Example | SR Appropriate? |
|--------|---------|----------------|
| Fundamental physics | `E = mc²`, `F = ma` | ✅ Excellent |
| Simple tabular data | Polynomial relationships | ✅ Very good |
| Noisy time series | Finance, weather | ⚠️ Limited |
| Images/Text | Vision, NLP | ❌ Unsuited |
| Complex multivariate (>10 vars) | Biomedical | ⚠️ Difficult |

---

## 📈 Comparisons

### Summary Table: SR vs Classical ML

| Criterion | SR (GP/MCTS) | EQL | Neural Networks | Random Forest | SVR |
|-----------|-------------|-----|-----------------|---------------|-----|
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ |
| **Performance (large data)** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Performance (small data)** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Extrapolation** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ |
| **Training speed** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Noise robustness** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 📖 Resources and References

### 📚 Key Articles

1. **La Cava et al. (2021)** - *"Contemporary Symbolic Regression Methods and their Relative Performance"* - Benchmark of 14 SR methods
2. **Makke & Chawla (2024)** - *"Interpretable Scientific Discovery with Symbolic Regression"* - Comprehensive review
3. **Martius & Lampert (2016)** - *"Extrapolation and learning equations"* - Original EQL paper
4. **Kommenda et al. (2024)** - *"SRBench++: A Living Benchmark Framework for Symbolic Regression"*

### 🔗 Useful Links

- [gplearn Documentation](https://gplearn.readthedocs.io/)
- [EQL Original Paper](https://arxiv.org/abs/1610.02995)
- [PySR (Julia backend)](https://github.com/MilesCranmer/PySR)
- [CERN Open Data](https://opendata.cern.ch/)
- [AIMS Senegal](https://aims-senegal.org/)

---

## 📝 Citation

If you use this code or results in your research, please cite:

```bibtex
@mastersthesis{fall2025symbolic,
  title={Symbolic Regression: A Methodological Framework for Interpretable Learning},
  author={Fall, Thiémokho},
  year={2025},
  school={African Institute for Mathematical Sciences (AIMS) - Senegal},
  type={Master's Thesis},
  url={https://github.com/thiefall/Thiemokho-Code-essay-AIMS-Senegal}
}
```

---

## 👥 Author

**Thiémokho Fall**  
🎓 Master Student @ AIMS Senegal  
📧 Email: [thiemokho.fall@aims-senegal.org](mailto:thiemokho.fall@aims-senegal.org)  
🔗 GitHub: [@thiefall](https://github.com/thiefall)  
🔗 LinkedIn: [www.linkedin.com/in/thiemokho-fall-6908ab1b9](www.linkedin.com/in/thiemokho-fall-6908ab1b9)

---

## 🙏 Acknowledgments

- **AIMS Senegal** for academic supervision
- **CERN Open Data** for access to particle physics data
- **Open Source Community** (scikit-learn, gplearn, TensorFlow, etc.)
- **Supervisors and peers** for constructive feedback

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Thiémokho Fall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🚧 Roadmap & Contributions

### 🎯 Future Developments

- [ ] Implement MCTS for SR (currently only GP is covered)
- [ ] Add benchmark on complete Feynman dataset
- [ ] Integrate PySR (Julia) for enhanced performance
- [ ] Interactive web interface (Streamlit/Gradio)
- [ ] Multivariate extension with automatic feature engineering
- [ ] Deep dive into EQL hyperparameter optimization

### 🤝 How to Contribute?

Contributions are welcome! Please:

1. Fork the project
2. Create a branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Contact & Support

Questions? Suggestions? Feel free to:

- 📧 Send me an email
- 🐛 Open an [Issue](https://github.com/thiefall/Thiemokho-Code-essay-AIMS-Senegal/issues)
- 💬 Start a [Discussion](https://github.com/thiefall/Thiemokho-Code-essay-AIMS-Senegal/discussions)

---
📝 Declaration

I, the undersigned, hereby declare that the work contained in this essay is my original work, and that any work done by others or by myself previously has been acknowledged and referenced accordingly
                                                                Thiemokho Fall
<div align="center">

### ⭐ If this project was helpful, please give it a star!

**Made with ❤️ at AIMS Senegal 🇸🇳**

</div>

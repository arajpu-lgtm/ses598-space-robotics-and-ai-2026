# Assignment 0: Least Squares and Probability Theory

## Overview

This foundational assignment introduces core mathematical concepts that underpin estimation and control methods used throughout the course. While **not graded** (0 points), completing this assignment demonstrates mastery of fundamental estimation theory and may be considered in final grade determinations if needed.

## Learning Objectives

By completing this assignment, students will:

1. **Apply Least Squares Estimation**: Use least squares methods to estimate parameters from noisy data
3. **Analyze Real-World Data**: Work with geological data to solve a practical estimation problem
4. **Quantify Uncertainty**: Compute and interpret estimation uncertainty
5. **Validate Models**: Assess model fit and residual analysis

## Problem Description

### The Pacific Plate Velocity Problem

You will estimate the velocity of the Pacific tectonic plate using volcanic island age and distance data. As the Pacific Plate moves over a stationary mantle hotspot (currently under Hawaii), it creates a chain of volcanic islands. By analyzing the age and distance of these islands from the hotspot, you can estimate the plate's velocity.

**Physical Model:**
```
distance = velocity × age + noise
```

Your task is to:
1. Estimate the plate velocity using least squares
2. Compute the uncertainty in your estimate
3. Validate your model against the data
4. Interpret your results in a geological context

## Assignment Materials

### 📄 Files in This Directory

1. **SES598_2026_Assignment0_least_squares_probability_theory.pdf**
   - Complete assignment description
   - Problem formulation
   - Deliverables and expectations

2. **volcanoes_data_ses598_2026.csv**
   - Dataset containing volcanic island information
   - Columns: island name, age (millions of years), distance from hotspot (km)
   - Ready to use with Python

3. **SES598_2026_notes_least_squares_MLE-1-1.pdf**
   - Lecture notes on least squares estimation
   - Maximum likelihood estimation theory
   
## Getting Started

### Prerequisites

**Software:**
- Python 3.8+ (recommended) 
- NumPy, Matplotlib, Pandas (for Python)
- Jupyter Notebook (optional but recommended)

**Mathematical Background:**
- Linear algebra (vectors, matrices)
- Basic probability and statistics
- Calculus (derivatives)

### Installation

#### Python Setup (Recommended)

```bash
# Install required packages
pip3 install numpy scipy matplotlib pandas jupyter

# Optional: Install for better plotting
pip3 install seaborn
```

#### Verify Installation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("✓ All packages installed successfully!")
```

### Quick Start

```python
# Load the data
import pandas as pd
data = pd.read_csv('volcanoes_data_ses598_2026.csv')

# Display first few rows
print(data.head())

# Your least squares implementation goes here...
```

## Assignment Structure

### Part 1: Data Exploration (Recommended)
- Load and visualize the volcano data
- Plot age vs. distance
- Identify any outliers or patterns

### Part 2: Least Squares Estimation
- Formulate the estimation problem
- Derive the least squares solution
- Implement the estimator
- Compute the velocity estimate

### Part 3: Uncertainty Quantification
- Compute estimation covariance
- Calculate confidence intervals
- Assess parameter uncertainty

### Part 4: Model Validation
- Compute residuals
- Plot residuals vs. predictions
- Assess goodness of fit (R², RMSE)
- Discuss model assumptions

### Part 5: Interpretation
- Interpret your velocity estimate
- Compare with published values (if available)
- Discuss limitations and assumptions
- Propose improvements

## Expected Deliverables

While not formally graded, you should produce:

1. **Code Implementation**
   - Clean, well-commented code
   - Reproducible analysis
   - Clear variable names

2. **Visualizations**
   - Data plot (age vs. distance)
   - Fitted model overlay
   - Residual plots
   - Uncertainty visualization

3. **Written Analysis** (brief)
   - Velocity estimate with units
   - Confidence interval
   - Model validation discussion
   - Physical interpretation


## Tips for Success

### Data Analysis
1. Always plot your data first
2. Check for outliers or anomalies
3. Verify units are consistent
4. Document any data preprocessing

### Least Squares Implementation
1. Start with the mathematical derivation
2. Verify matrix dimensions
3. Implement manually first, and then explore numpy.linalg.lstsq() to compare 
4. Check for numerical stability

### Uncertainty Analysis
1. Don't forget to estimate noise variance
2. Propagate uncertainty properly
3. Report confidence intervals with your estimate
4. Visualize uncertainty on your plots

### Common Pitfalls to Avoid
- ❌ Forgetting to include intercept term
- ❌ Mixing units (years vs. millions of years)
- ❌ Not validating assumptions
- ❌ Ignoring outliers without justification
- ❌ Over-interpreting results

## Example Code Structure

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data
data = pd.read_csv('volcanoes_data_ses598_2026.csv')
age = data['age'].values  # Example column name
distance = data['distance'].values  # Example column name

# 2. Formulate problem
# y = H * x + noise
# where y = distance, x = [velocity, offset]^T

# 3. Least squares solution
# x_hat = (H^T H)^-1 H^T y

# 4. Compute covariance
# P = sigma^2 (H^T H)^-1

# 5. Validate model
# residuals = y - H * x_hat
# Plot residuals, compute R^2, etc.

# 6. Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(age, distance, label='Data')
# Plot your fitted line
plt.xlabel('Age (millions of years)')
plt.ylabel('Distance from hotspot (km)')
plt.legend()
plt.grid(True)
plt.show()
```

## Theoretical Background

### Least Squares Estimation

Given a linear model: **y = Hx + ν**

Where:
- **y**: measurements (n × 1)
- **H**: observation matrix (n × m)
- **x**: parameters to estimate (m × 1)
- **ν**: zero-mean Gaussian noise

The least squares estimate minimizes: ||**y** - **Hx**||²

**Solution:** **x̂** = (**H**ᵀ**H**)⁻¹**H**ᵀ**y**

**Covariance:** **P** = σ²(**H**ᵀ**H**)⁻¹

where σ² is the noise variance (estimated from residuals)

### Maximum Likelihood Estimation

If noise is Gaussian: **ν** ~ N(0, σ²**I**)

Then the MLE is equivalent to the least squares estimate!

This connects probability theory to optimization.

## Resources

### Course Materials
- Lecture slides (see course website)
- Lecture notes PDF in this folder
- Course samples in `samples/` directory

### External Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [SciPy Least Squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

### Recommended Books

**Introduction to Linear Algebra (Fifth Edition)**
- Author: Gilbert Strang
- A comprehensive guide to linear algebra fundamentals with clear explanations and practical applications.

**Optimal State Estimation: Kalman, H Infinity, and Nonlinear Approaches**
- Author: Dan Simon
- A comprehensive guide to state estimation techniques with practical applications in navigation and control systems.

**Pattern Recognition and Machine Learning**
- Author: Christopher M. Bishop
- The definitive text on modern pattern recognition methods with a focus on Bayesian techniques and machine learning algorithms.


## Frequently Asked Questions

### Q: Do I need to submit this assignment?
**A:** No formal submission required, but keeping your work organized is recommended for your own learning and potential future reference.

### Q: How much time should I spend on this?
**A:** 2-4 hours for implementation and analysis. Focus on understanding concepts over perfection.

### Q: Can I use built-in least squares functions?
**A:** Yes! Using `numpy.linalg.lstsq()` or `scipy.optimize.least_squares()` is fine. Understanding the math is what matters.

### Q: What if I get a different velocity than expected?
**A:** That's okay! Focus on your methodology. If you're far off, check your units and formulation.

### Q: Should I include an intercept term?
**A:** Yes, generally a good idea. The model becomes: distance = velocity × age + offset

### Q: How do I handle outliers?
**A:** You can identify outliers using residual analysis. Discuss them in your write-up but be careful about blindly removing data.

### Q: Can I work in groups?
**A:** Discussion is encouraged, but everyone should implement their own solution.

## Assessment Criteria (Self-Assessment)

While not graded, evaluate your work on:

- ✅ **Correctness**: Does your implementation produce reasonable results?
- ✅ **Code Quality**: Is your code clean, documented, and reproducible?
- ✅ **Analysis**: Did you validate your model and discuss results?
- ✅ **Visualization**: Are your plots clear and informative?
- ✅ **Understanding**: Can you explain your approach and results?

## Need Help?

### Before Asking:
1. Review the lecture notes PDF in this folder
2. Check the course textbook/slides
3. Try debugging with print statements
4. Verify your data loaded correctly

### Where to Get Help:
- Office hours (Fri, 11a-12p, Zoom and Walton Center)
- Canvas and Slack
  
## Why This Assignment Matters

Estimation theory is fundamental to robotics and AI:

- **State Estimation**: Where is my robot? (Kalman filters, particle filters)
- **Mapping**: What does the environment look like? (SLAM, structure from motion)
- **Learning**: What are the model parameters? (Neural networks, system ID)
- **Control**: What control minimizes cost? (LQR, MPC)

By mastering least squares and probability theory, you build intuition for how robots perceive, model, and interact with the world.

---

## Summary

**Assignment Type:** Optional, not graded  
**Estimated Time:** 2-4 hours  
**Prerequisites:** Basic Python/MATLAB, linear algebra, probability  
**Purpose:** Build foundations for state estimation and optimal control

**Key Takeaways:**
- Least squares minimizes squared error
- MLE connects probability to optimization
- Uncertainty quantification is crucial
- Model validation reveals assumptions
- Real data is noisy and imperfect

---

**Ready to start?** Open the assignment PDF and dive into the volcanic data!

**Questions?** Review the lecture notes PDF or reach out during office hours.

## Acknowledgments

This assignment uses publicly available geological data on Hawaiian volcanic island ages and positions. The Pacific Plate velocity estimation problem is a fun application of least squares estimation in Earth sciences.

**Course:** SES 598 Space Robotics and AI  
**Institution:** Arizona State University  
**Instructor:** Jnaneshwar Das

---

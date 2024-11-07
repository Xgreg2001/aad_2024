import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Part (a):
rng = np.random.default_rng(10)
x1 = rng.uniform(0, 1, size=100)
x2 = 0.5 * x1 + rng.normal(size=100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size=100)

# Write out the form of the linear model and regression coefficients
print("Part (a):")
print("The linear model is:")
print("Y = β0 + β1 * X1 + β2 * X2 + ε")
print("Where:")
print("β0 = 2")
print("β1 = 2")
print("β2 = 0.3")
print()

# Part (b):
corr_x1_x2 = np.corrcoef(x1, x2)[0, 1]
print("Part (b):")
print(f"The correlation between x1 and x2 is: {corr_x1_x2:.4f}")

# Create scatterplot of x1 vs x2
plt.scatter(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatterplot of x1 vs x2')
plt.savefig('scatterplot.pdf')
plt.close()

# Part (c):
X = np.column_stack((x1, x2))
X = sm.add_constant(X)  # Adds an intercept term
model = sm.OLS(y, X).fit()
print("Part (c):")
print(model.summary())
beta_hat = model.params
print(f"Estimated coefficients:")
print(f"β̂0 (Intercept): {beta_hat[0]:.4f}")
print(f"β̂1: {beta_hat[1]:.4f}")
print()

# Hypothesis tests
p_value_beta1 = model.pvalues[1]
p_value_beta2 = model.pvalues[2]
print(f"P-value for β1: {p_value_beta1:.4f}")
print(f"P-value for β2: {p_value_beta2:.4f}")
print()

# Part (d):
X1 = sm.add_constant(x1)
model_x1 = sm.OLS(y, X1).fit()
print("Part (d):")
print(model_x1.summary())
beta_hat_x1 = model_x1.params
print(f"Estimated coefficients:")
print(f"β̂0 (Intercept): {beta_hat_x1[0]:.4f}")
print(f"β̂1: {beta_hat_x1[1]:.4f}")
p_value_beta1_x1 = model_x1.pvalues[1]
print(f"P-value for β1: {p_value_beta1_x1:.4f}")
print()

# Part (e):
X2 = sm.add_constant(x2)
model_x2 = sm.OLS(y, X2).fit()
print("Part (e):")
print(model_x2.summary())
beta_hat_x2 = model_x2.params
print(f"Estimated coefficients:")
print(f"β̂0 (Intercept): {beta_hat_x2[0]:.4f}")
print(f"β̂1: {beta_hat_x2[1]:.4f}")
p_value_beta1_x2 = model_x2.pvalues[1]
print(f"P-value for β1: {p_value_beta1_x2:.4f}")
print()

# Part (g):
x1 = np.concatenate([x1, [0.1]])
x2 = np.concatenate([x2, [0.8]])
y = np.concatenate([y, [6]])

# Re-fit the model from part (c)
X = np.column_stack((x1, x2))
X = sm.add_constant(X)
model_new = sm.OLS(y, X).fit()
print("Part (g):")
print("Model with x1 and x2 after adding new observation:")
print(model_new.summary())
print()

# Re-fit the model from part (d)
X1 = sm.add_constant(x1)
model_x1_new = sm.OLS(y, X1).fit()
print("Model with x1 only after adding new observation:")
print(model_x1_new.summary())
print()

# Re-fit the model from part (e)
X2 = sm.add_constant(x2)
model_x2_new = sm.OLS(y, X2).fit()
print("Model with x2 only after adding new observation:")
print(model_x2_new.summary())
print()

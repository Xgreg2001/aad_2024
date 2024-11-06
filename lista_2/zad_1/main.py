# Exercise: Simulated Linear Regression in Python

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Set seed for reproducibility
np.random.seed(1)

# Part (a): Generate a vector x of 100 observations from N(0, 1)
x = np.random.normal(0, 1, 100)

# Part (b): Generate a vector eps of 100 observations from N(0, 0.25)
eps = np.random.normal(0, 0.5, 100)  # Standard deviation is sqrt(0.25) = 0.5

# Part (c): Generate y according to the model Y = -1 + 0.5X + eps
y = -1 + 0.5 * x + eps

# Output the length of y and values of beta_0 and beta_1
print("Length of y:", len(y))
print("Beta_0 (β0): -1")
print("Beta_1 (β1): 0.5")

# Part (d): Create a scatterplot of x and y
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatterplot of x vs y')
plt.savefig('scatterplot_xy.pdf')
plt.close()

# Part (e): Fit a least squares linear model to predict y using x
X = sm.add_constant(x)  # Adds an intercept term to the model
model = sm.OLS(y, X).fit()
print(model.summary())
print(f"Estimated Beta_0 (β̂0): {model.params[0]:.4f}")
print(f"Estimated Beta_1 (β̂1): {model.params[1]:.4f}")
print("The estimated coefficients are close to the true values of -1 and 0.5.")

# Part (f): Display the least squares line and the population regression line
# Sorting x for plotting lines
sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_pred = model.predict(X)
y_pred_sorted = y_pred[sort_idx]
y_pop = -1 + 0.5 * x_sorted  # Population regression line

plt.scatter(x, y, label='Data')
plt.plot(x_sorted, y_pred_sorted, color='red', label='Fitted Regression Line')
plt.plot(x_sorted, y_pop, color='green', label='Population Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatterplot with Regression Lines')
plt.legend()
plt.savefig('scatterplot_regression_lines.pdf')
plt.close()

# Part (g): Fit a polynomial regression model predicting y using x and x^2
x2 = x ** 2
X_poly = np.column_stack((x, x2))
X_poly = sm.add_constant(X_poly)
model_poly = sm.OLS(y, X_poly).fit()
print(model_poly.summary())

# Part (h): Repeat (a)-(f) with less noise (variance decreased to 0.05)


def simulate_linear_regression(noise_variance, seed=1):
    np.random.seed(seed)
    x = np.random.normal(0, 1, 100)
    eps = np.random.normal(0, np.sqrt(noise_variance), 100)
    y = -1 + 0.5 * x + eps

    # Fit linear regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Plotting
    y_pred = model.predict(X)
    y_pop = -1 + 0.5 * x

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    y_pop_sorted = y_pop[sort_idx]

    plt.scatter(x, y, label='Data')
    plt.plot(x_sorted, y_pred_sorted, color='red',
             label='Fitted Regression Line')
    plt.plot(x_sorted, y_pop_sorted, color='green',
             label='Population Regression Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(
        f'Scatterplot with Regression Lines (Noise Variance={noise_variance})')
    plt.legend()
    plt.savefig(
        f'scatterplot_regression_lines_noise_variance_{noise_variance:.2f}.pdf')
    plt.close()

    print(model.summary())
    return model


print("\nResults with Less Noise (Variance = 0.05):")
model_less_noise = simulate_linear_regression(noise_variance=0.05)

# Part (i): Repeat (a)-(f) with more noise (variance increased to 1)
print("\nResults with More Noise (Variance = 1):")
model_more_noise = simulate_linear_regression(noise_variance=1)

# Part (j): Confidence intervals for β0 and β1
print("\nConfidence intervals for the original data set:")
print(model.conf_int())

print("\nConfidence intervals for the less noisy data set:")
print(model_less_noise.conf_int())

print("\nConfidence intervals for the more noisy data set:")
print(model_more_noise.conf_int())

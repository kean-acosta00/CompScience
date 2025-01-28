import numpy as np 
import matplotlib.pyplot as plt

# DATAS 
hrs_practiced = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
proficiency_lvl = np.array([2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10])

def linear_regression(hrs_practiced, proficiency_lvl):
    hp_mean = np.mean(hrs_practiced)
    pl_mean = np.mean(proficiency_lvl)
    # LINEAR REGRESSION FORMULA
    slope = sum((hrs_practiced - hp_mean) * (proficiency_lvl - pl_mean)) / sum((hrs_practiced - hp_mean) ** 2)
    # Y-INTERCEPT FORMULA
    b = pl_mean - (slope * hp_mean)

    return slope, b

# PERFORM LINEAR REGRESSION
slope, b = linear_regression(hrs_practiced, proficiency_lvl)

hp_line = np.linspace(min(hrs_practiced), max(hrs_practiced), 100)
pl_line = slope * hp_line + b

def plot_regression(hrs_practiced, proficiency_lvl):
    plt.scatter(hrs_practiced, proficiency_lvl, color='blue', label='Data Points')
    plt.plot(hp_line, pl_line, color='red', label='Regression Line')
    plt.xlabel('Hours Practiced')
    plt.ylabel('Proficiency Level')
    plt.title('Linear Regression Model')
    plt.legend()
    plt.show()

# Display results
print(f"Slope: {slope}")
print(f"Y-Intercept: {b}")

plot_regression(hrs_practiced, proficiency_lvl)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

n_simulations = 40000
n_clients_per_grade = 200
pds = np.linspace(0.02, 0.20, 10)
rho = 0.1
target_percentile = 95

np.random.seed(42)

S = np.random.standard_normal(n_simulations)[:, np.newaxis]

term_1 = norm.ppf(pds)                 
term_2 = np.sqrt(rho) * S               
denominator = np.sqrt(1 - rho)

conditional_pds = norm.cdf((term_1 - term_2) / denominator)

defaults_per_grade = np.random.binomial(n=n_clients_per_grade, p=conditional_pds)

portfolio_defaults = np.sum(defaults_per_grade, axis=1)

predicted_defaults = np.percentile(portfolio_defaults, target_percentile)

print(f"95th Percentile of Predicted Defaults: {predicted_defaults:.0f}")

plt.figure(figsize=(10, 6))
plt.hist(portfolio_defaults, bins=50, density=False, alpha=0.75, color='steelblue', edgecolor='black')
plt.axvline(predicted_defaults, color='red', linestyle='dashed', linewidth=2, 
            label=f'95th Percentile: {predicted_defaults:.0f} defaults')

plt.title('Simulated Portfolio Default Distribution (Vasicek ASRF Model)')
plt.xlabel('Total Number of Defaults in Portfolio')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

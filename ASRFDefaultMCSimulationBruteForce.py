import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

n_simulations = 40000
n_grades = 10
n_clients_per_grade = 10000 
pds = np.linspace(0.02, 0.20, n_grades)
rho = 0.1
target_percentile = 95
np.random.seed(42)

start_time = time.time()

thresholds = norm.ppf(pds)
client_thresholds = np.repeat(thresholds, n_clients_per_grade)
n_total_clients = len(client_thresholds)

sqrt_rho = np.sqrt(rho)
sqrt_1_rho = np.sqrt(1 - rho)
portfolio_defaults = np.zeros(n_simulations)

for i in range(n_simulations):
    S = np.random.standard_normal()
    Z = np.random.standard_normal(n_total_clients)
    A = sqrt_rho * S + sqrt_1_rho * Z
    defaults = A < client_thresholds
    portfolio_defaults[i] = np.sum(defaults)
    
    if (i + 1) % 10000 == 0:
        print(f"Completed {i + 1} / {n_simulations} simulations...")

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.\n")


predicted_defaults = np.percentile(portfolio_defaults, target_percentile)
print(f"95th Percentile of Predicted Defaults: {predicted_defaults:.0f}")

plt.figure(figsize=(10, 6))
plt.hist(portfolio_defaults, bins=50, density=False, alpha=0.75, color='steelblue', edgecolor='black')
plt.axvline(predicted_defaults, color='red', linestyle='dashed', linewidth=2, 
            label=f'95th Percentile: {predicted_defaults:.0f} defaults')

plt.title('Simulated Portfolio Default Distribution (Brute Force Asset Approach)')
plt.xlabel('Total Number of Defaults in Portfolio')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

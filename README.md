# PDESolvers ૮₍  ˶•⤙•˶ ₎ა

A Python package for solving partial differential equations (PDEs), including the one-dimensional heat equation and the Black-Scholes equation, using numerical methods such as explicit and Crank-Nicolson finite difference schemes. Features include built-in plotting, benchmarking and support for financial applications such as option pricing.

## 📦 Installation
The pdesolvers package can be installed using pip. To install the package, run the following command:
```
pip install pdesolvers
```
Updating the package to its latest version can be done with:
```
pip install --upgrade pdesolvers
```

## 🧩 Supported Features
### Solvers
- ✅ **Explicit method**
- ✅ **Crank-Nicolson method**

### Equations
- ✅ **1D Heat Equation**
- ✅ **Black-Scholes Equation for vanilla European options**

### Pricing Methods
- ✅ **Monte Carlo Pricing**
- ✅ **Analytical Black-Scholes formula**

## 📁 Project Structure
```plaintext
PDESolvers/
├── pdesolvers/                # Main Python package
│   ├── enums/                 # Enum definitions (e.g., option types, greeks)
│   ├── optionspricing/        # Modules for Monte Carlo and Black-Scholes pricing
│   ├── pdes/                  # PDE definitions (HeatEquation, BlackScholesEquation, etc.)
│   ├── solution/              # Solution classes (Solution1D, SolutionBlackScholes, etc.)
│   ├── solvers/               # Explicit, Crank-Nicolson, etc.
│   ├── tests/                 # Unit tests for Python components
│   ├── utils/                 # Helper functions
│   ├── __init__.py            # Makes it a package
├── GPUSolver/                 # GPU-accelerated module
│   ├── cpu/                   # CPU-side implementations (C++)
│   ├── gpu/                   # CUDA kernels and GPU logic
│   ├── tests/                 # Tests for GPU and C++ logic
```

## 📊 Export Options
Use **_export=True_** flags in plotting functions or benchmarking methods to export:
- **PDF**: Save plots as PDF files.
- **CSV**: Save benchmark results as CSV files.

## 🚀 Usage
To use the package, you can import the desired modules and classes and create an instance of the solvers.

### Example usage of the 1D heat equation solver
```python
from pdesolvers import HeatEquation, Heat1DExplicitSolver, Heat1DCNSolver
import numpy as np

equation = (HeatEquation(1, 100,30,10000, 0.01)
            .set_initial_temp(lambda x: np.sin(np.pi * x) + 5)
            .set_left_boundary_temp(lambda t: 20 * np.sin(np.pi * t) + 5)
            .set_right_boundary_temp(lambda t: t + 5))

solution1 = Heat1DCNSolver(equation).solve()
solution2 = Heat1DExplicitSolver(equation).solve()

result = solution1.get_result()
solution1.plot()
```

### Example usage of the Black-Scholes equation solver
```python
from pdesolvers import BlackScholesEquation, BlackScholesExplicitSolver, BlackScholesCNSolver, OptionType, Greeks

equation = BlackScholesEquation(OptionType.EUROPEAN_CALL, 300, 295, 0.05, 0.2, 1, 100, 10000)

solution1 = BlackScholesExplicitSolver(equation).solve()
solution2 = BlackScholesCNSolver(equation).solve()

solution1.plot()
solution1.plot_greek(Greeks.GAMMA)
solution2.get_execution_time()
```

### Example usage of Monte Carlo Pricing
```python
from pdesolvers import MonteCarloPricing, OptionType

pricing = MonteCarloPricing(OptionType.EUROPEAN_CALL, 300, 290, 0.05, 0.2, 1, 365, 1000, 78)
option_price = pricing.get_monte_carlo_option_price()

pricing.plot_price_paths()
pricing.plot_distribution_of_payoff()
pricing.plot_distribution_of_final_prices()
```

### Example usage of Analytical Black-Scholes formula
```python
from pdesolvers import BlackScholesFormula, OptionType

pricing = BlackScholesFormula(OptionType.EUROPEAN_CALL, 300, 290, 0.05, 0.2, 1)
option_price = pricing.get_black_scholes_merton_price()
```

### Using Real Historical Data
```python
from pdesolvers import HistoricalStockData, MonteCarloPricing, OptionType

ticker = 'NVDA'

historical_data = HistoricalStockData(ticker)
historical_data.fetch_stock_data( "2024-02-28","2025-02-28")

sigma, mu = historical_data.estimate_metrics()
initial_price = historical_data.get_initial_stock_price()
closing_prices = historical_data.get_closing_prices()

pricing = MonteCarloPricing(OptionType.EUROPEAN_CALL, initial_price, 160, mu, sigma, 1, len(closing_prices), 1000, 78)

pricing.plot_price_paths(export=True)
```
> 📝 **Note:** You don't necessarily need to use the HistoricalStockData class — you're free to use raw yfinance data directly.
Use the built-in tools only if you want to estimate metrics like volatility or mean return.

### 📊 Comparing Interpolated Grid Solutions
```python
from pdesolvers import BlackScholesEquation, BlackScholesExplicitSolver, BlackScholesCNSolver, OptionType

equation1 = BlackScholesEquation(OptionType.EUROPEAN_CALL, S_max=300, K=100, r=0.05, sigma=0.2, expiry=1, s_nodes=100, t_nodes=1000)
equation2 = BlackScholesEquation(OptionType.EUROPEAN_CALL, S_max=300, K=100, r=0.05, sigma=0.2, expiry=1)

solution1 = BlackScholesExplicitSolver(equation1).solve()
solution2 = BlackScholesCNSolver(equation1).solve()

error = solution1 - solution2
```

### 📊 Additional Benchmarks
```python
from pdesolvers import MonteCarloPricing, BlackScholesFormula, OptionType

num_simulations_list = [ 20, 50, 100, 250, 500, 1000, 2500]

pricing_1 = BlackScholesFormula(OptionType.EUROPEAN_CALL, 300, 290, 0.05, 0.2, 1)
pricing_2 = MonteCarloPricing(OptionType.EUROPEAN_CALL, 300, 290, 0.05, 0.2, 1, 365, 1000000, 78)

bs_price = pricing_1.get_black_scholes_merton_price()
monte_carlo_price = pricing_2.get_monte_carlo_option_price()

pricing_2.get_benchmark_errors(bs_price, num_simulations_list=num_simulations_list)
pricing_2.plot_convergence_analysis(bs_price, num_simulations_list=num_simulations_list, export=True)
```
> 📝 **Note:** The export flag used in the example above will save the plot as a PDF file in the current working directory.

## 🧠 Limitations
- The package currently supports only one-dimensional PDEs.
- Currently limited to vanilla European options.
- GPU acceleration only implemented for finite difference methods. 

> 📝 **Note:** The Python and C++/CUDA libraries are currently developed as separate components and are not integrated. The Python library can be used independently via PyPI, while the GPU-accelerated solvers are available as a standalone C++/CUDA project.

## 🔒 License
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE.md) file for details.
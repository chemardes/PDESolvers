import numpy as np
import matplotlib.pyplot as plt
import time
import logging

from pdesolvers.enums.enums import OptionType

logging.basicConfig(
    level = logging.INFO,
    format = "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

class MonteCarloPricing:

    def __init__(self, option_type: OptionType, S0, strike_price, mu, sigma, T, time_steps, sim, seed=None):
        """
        Initialize the Geometric Brownian Motion model with the given parameters.

        Parameters:
        - S0: Initial stock price
        - mu: Drift coefficient (expected return)
        - sigma: Volatility coefficient (standard deviation of returns)
        - T: Time period for the simulation (in years)
        - time_steps: Number of time steps in the simulation
        - sim: Number of simulations to run
        """

        self.__option_type = option_type
        self.__S0 = S0
        self.__strike_price = strike_price
        self.__mu = mu
        self.__sigma = sigma
        self.__T = T
        self.__time_steps = time_steps
        self.__sim = sim
        self.__seed = seed
        self.__S = None
        self.__payoff = None
        self.__run = False
        self.__duration = 0.0

    def get_monte_carlo_option_price(self):
        """
        Calculates the price of the option based on the monte carlo simulations
        :return: the option price
        """

        logging.info(f"Starting {self.__class__.__name__}")
        logging.info(f"Calculating the option price with {self.__sim} simulations")

        S = self.simulate_gbm()
        self.__run = True

        if self.__option_type == OptionType.EUROPEAN_CALL:
            self.__payoff = np.maximum(S[:, -1] - self.__strike_price, 0)
        elif self.__option_type == OptionType.EUROPEAN_PUT:
            self.__payoff = np.maximum(self.__strike_price - S[:, -1], 0)
        else:
            raise ValueError(f'Unsupported Option Type: {self.__option_type}')

        option_price = np.exp(-self.__mu * self.__T) * np.mean(self.__payoff)

        logging.info(f"Option price calculated successfully with {self.__class__.__name__}.")

        return option_price

    def simulate_gbm(self):
        """
        Simulate the Geometric Brownian Motion for the given parameters.

        This method calculates the stock prices at each time step for each simulation.
        """

        logging.info("Simulating the Geometric Brownian motion...")

        if self.__seed:
            np.random.seed(self.__seed)

        start = time.perf_counter()

        t = self.__generate_grid()
        dt = t[1] - t[0]

        B = np.zeros((self.__sim, self.__time_steps))
        S = np.zeros((self.__sim, self.__time_steps))

        # for all simulations at t = 0
        S[:,0] = self.__S0
        Z = np.random.normal(0, 1, (self.__sim, self.__time_steps))

        for i in range(self.__sim):
            for j in range (1, self.__time_steps):
                # updates brownian motion
                B[i,j] = B[i,j-1] + np.sqrt(dt) * Z[i,j-1]
                # calculates stock price based on the incremental difference
                S[i,j] = S[i, j-1] * np.exp((self.__mu - 0.5*self.__sigma**2)*dt + self.__sigma * (B[i, j] - B[i, j - 1]))

        end = time.perf_counter()
        self.__duration = end - start

        logging.info(f"Simulation completed in {self.__duration} seconds.")

        self.__S = S
        return self.__S

    def get_benchmark_errors(self, analytical_solution, num_simulations_list, export=False):
        """
        Calculates the absolute errors of each simulation count

        :param analytical_solution: the analytical solution of the black-scholes formula
        :param num_simulations_list: list of number of simulations
        :param export: boolean flag to indicate whether the results should be exported as csv
        :return: list of absolute errors corresponding to the number of simulations
        """

        prices = []
        errors = []

        original_sim = self.__sim

        for sim in num_simulations_list:
            self.__sim = sim
            self.__run = False

            print(f"Running simulation for Simulation Count: {sim}")

            sim_price = self.get_monte_carlo_option_price()
            prices.append(sim_price)
            error = abs(sim_price - analytical_solution)
            errors.append(error)

        self.__sim = original_sim
        self.__run = False

        if export:
            np.savetxt("benchmark_errors.csv", np.column_stack((num_simulations_list, errors)),
                       delimiter=",", header="Simulations,Error", comments="")

        return errors

    def __generate_grid(self):
        """
        Generate a time grid from 0 to T with `time_steps` intervals.

        Returns:
        - A numpy array representing the time grid.
        """

        return np.linspace(0, self.__T, self.__time_steps)

    def plot_price_paths(self, closing_prices=None, export=False):
        """
        Plot the simulated stock prices for all simulations.
        """

        if not self.__run:
            raise RuntimeError("Plots cannot be generated because the simulation has not been run yet.")

        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.size'] = 18

        t = self.__generate_grid()

        fig = plt.figure(figsize=(10,6))
        for i in range(np.min([100, self.__sim])):
            plt.plot(t, self.__S[i], color='grey', alpha=0.3)

        plt.title("Simulated Geometric Brownian Motion")
        plt.xlabel("Time (Years)")
        plt.ylabel("Stock Price")

        if closing_prices is not None:
            if len(closing_prices) != len(t):
                raise ValueError("Length of closing prices does not match the number of time steps in the simulation.")

            plt.plot(t , closing_prices, color='red')

        if export:
            plt.savefig("monte_carlo_prices.pdf", format="pdf", bbox_inches="tight")

        plt.show()

    def plot_distribution_of_final_prices(self, export=False):
        """
        Plots the distribution of final prices at expiry
        :param export: boolean flag to indicate whether the plot should be exported as pdf
        :return:
        """

        if not self.__run:
            raise RuntimeError("Plots cannot be generated because the simulation has not been run yet.")

        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.size'] = 18

        final_prices = self.__S[:, -1]

        plt.figure(figsize=(10, 6))
        plt.hist(final_prices, bins=50, edgecolor='darkblue', alpha=0.5, color='blue', density=True)
        plt.title('Distribution of Final Stock Prices at Maturity')
        plt.xlabel('Stock Price')
        plt.ylabel('Frequency Density')

        if export:
            plt.savefig("monte_carlo_prices.pdf", format="pdf", bbox_inches="tight", pad_inches=0.2)

        plt.show()

    def plot_distribution_of_payoff(self, export=False):
        """
        Plots the distribution of payoff at expiru
        :param export: boolean flag to indicate whether the plot should be exported as pdf
        :return: plot
        """
        if not self.__run:
            raise RuntimeError("Plots cannot be generated because the simulation has not been run yet.")

        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.size'] = 18

        plt.figure(figsize=(10, 6))
        plt.hist(self.__payoff, bins=50, edgecolor='darkblue', alpha=0.5, color='blue', density=True)
        plt.title('Distribution of Payoff')
        plt.xlabel('Payoff')
        plt.ylabel('Frequency Density')

        if export:
            plt.savefig("monte_carlo_payoff.pdf", format="pdf", bbox_inches="tight", pad_inches=0.2)

        plt.show()

    def plot_convergence_analysis(self, analytical_solution, num_simulations_list=None, export=False):
        """

        :param analytical_solution: the analytical solution of the black-scholes formula
        :param num_simulations_list: list of number of simulations
        :param export: boolean flag to indicate whether the plot should be exported as pdf
        :return: plot
        """
        if num_simulations_list is None:
            raise ValueError("Number of simulations need to be defined.")

        errors = self.get_benchmark_errors(analytical_solution, num_simulations_list, export=export)

        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.size'] = 18

        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(num_simulations_list, errors, 'bo-', linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'Monte Carlo Error Convergence (Benchmark: {analytical_solution:.4f})')

        if export:
            plt.savefig("monte_carlo_convergence.pdf", format="pdf",
                    bbox_inches="tight", pad_inches=0.2)

        plt.show()

    def get_execution_time(self):
        if not self.__run:
            raise RuntimeError("Execution time is not available because the simulation has not been run yet.")

        return self.__duration
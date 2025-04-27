import pytest
import numpy as np
from unittest.mock import patch

from pdesolvers.optionspricing.monte_carlo import MonteCarloPricing
from pdesolvers.optionspricing.black_scholes_formula import BlackScholesFormula
from pdesolvers.enums.enums import OptionType, Greeks

@pytest.fixture
def mc_pricing_params():
    return {
        'S0': 100.0,
        'strike_price': 100.0,
        'mu': 0.05,
        'sigma': 0.2,
        'T': 1.0,
        'time_steps': 252,
        'sim': 5
    }

@pytest.fixture
def bs_pricing_params():
    return {
        'S0': 100.0,
        'strike_price': 100.0,
        'r': 0.05,
        'sigma': 0.2,
        'expiry': 1.0
    }

class TestMonteCarlo:

    @patch('pdesolvers.optionspricing.monte_carlo.MonteCarloPricing.simulate_gbm')
    def test_get_european_call_option_price(self, mock_simulate_gbm, mc_pricing_params):
        mock_price_array = np.zeros((5, mc_pricing_params['time_steps']))
        mock_price_array[:, -1] = np.array([110, 120, 90, 105, 115])

        mock_simulate_gbm.return_value = mock_price_array

        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_CALL, **mc_pricing_params
        )

        mock_payoffs = np.maximum(np.array([110, 120, 90, 105, 115]) - mc_pricing_params['strike_price'], 0)
        expected_price = np.exp(-mc_pricing_params['mu'] * mc_pricing_params['T']) * np.mean(mock_payoffs)

        actual_price = test_mc.get_monte_carlo_option_price()

        assert actual_price == expected_price

    @patch('pdesolvers.optionspricing.monte_carlo.MonteCarloPricing.simulate_gbm')
    def test_get_european_put_option_price(self, mock_simulate_gbm, mc_pricing_params):
        mock_price_array = np.zeros((5, mc_pricing_params['time_steps']))
        mock_price_array[:, -1] = np.array([110, 120, 90, 105, 115])

        mock_simulate_gbm.return_value = mock_price_array

        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_PUT, **mc_pricing_params
        )

        mock_payoffs = np.maximum(mc_pricing_params['strike_price'] - np.array([110, 120, 90, 105, 115]), 0)
        expected_price = np.exp(-mc_pricing_params['mu'] * mc_pricing_params['T']) * np.mean(mock_payoffs)

        actual_price = test_mc.get_monte_carlo_option_price()

        assert actual_price == expected_price

    def test_get_invalid_option_price(self, mc_pricing_params):

        test_mc = MonteCarloPricing(
            Greeks.THETA, **mc_pricing_params
        )

        with pytest.raises(ValueError, match="Unsupported Option Type: Greeks.THETA"):
            test_mc.get_monte_carlo_option_price()

    def test_check_simulate_gbm_results_shape(self, mc_pricing_params):

        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_PUT, **mc_pricing_params
        )

        results = test_mc.simulate_gbm()

        assert results.shape == (mc_pricing_params['sim'], mc_pricing_params['time_steps'])

    def test_check_simulate_gbm_initial_values(self, mc_pricing_params):
        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_CALL, **mc_pricing_params
        )

        results = test_mc.simulate_gbm()

        assert (results[:,0] == mc_pricing_params['S0']).all()

    def test_check_get_execution_time_raises_error_when_no_simulation_is_run(self, mc_pricing_params):

        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_CALL, **mc_pricing_params
        )

        with pytest.raises(RuntimeError, match="Execution time is not available because the simulation has not been run yet."):
            test_mc.get_execution_time()

    def test_check_plot_payoff_raises_error_when_no_simulation_is_run(self, mc_pricing_params):

        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_CALL, **mc_pricing_params
        )

        with pytest.raises(RuntimeError, match="Plots cannot be generated because the simulation has not been run yet."):
            test_mc.plot_distribution_of_payoff()

    def test_check_plot_price_distribution_raises_error_when_no_simulation_is_run(self, mc_pricing_params):

        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_CALL, **mc_pricing_params
        )

        with pytest.raises(RuntimeError, match="Plots cannot be generated because the simulation has not been run yet."):
            test_mc.plot_distribution_of_final_prices()

    def test_check_plot_price_paths_raises_error_when_no_simulation_is_run(self, mc_pricing_params):

        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_CALL, **mc_pricing_params
        )

        with pytest.raises(RuntimeError, match="Plots cannot be generated because the simulation has not been run yet."):
            test_mc.plot_price_paths()

    def test_check_plot_convergence_raises_error_when_no_list_supplied(self, mc_pricing_params, bs_pricing_params):

        test_mc = MonteCarloPricing(
            OptionType.EUROPEAN_CALL, **mc_pricing_params
        )

        results = test_mc.simulate_gbm()

        price = BlackScholesFormula(OptionType.EUROPEAN_CALL, **bs_pricing_params)

        with pytest.raises(ValueError, match="Number of simulations need to be defined."):
            test_mc.plot_convergence_analysis(price)

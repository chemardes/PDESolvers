import pytest
import numpy as np

import pdesolvers.pdes.black_scholes as bse
import pdesolvers.solvers.black_scholes_solvers as solver
import pdesolvers.utils.utility as utility

class TestBlackScholesSolvers:

    def setup_method(self):
        self.equation = bse.BlackScholesEquation('call', 300, 1, 0.2, 0.05, 100, 100, 2000)

    # explicit method tests

    def test_check_lower_boundary_for_call_explicit(self):
        result = solver.BlackScholesExplicitSolver(self.equation).solve().get_result()
        assert np.all(result[0,:]) == 0

    def test_check_terminal_condition_for_call_explicit(self):
        result = solver.BlackScholesExplicitSolver(self.equation).solve().get_result()

        test_asset_grid = self.equation.generate_asset_grid()
        test_strike_price = self.equation.strike_price
        expected_payoff = np.maximum(test_asset_grid - test_strike_price, 0)

        assert np.array_equal(result[:, -1], expected_payoff)

    def test_check_terminal_condition_for_put_explicit(self):
        self.equation.option_type = 'put'
        result = solver.BlackScholesExplicitSolver(self.equation).solve().get_result()

        test_asset_grid = self.equation.generate_asset_grid()
        test_strike_price = self.equation.strike_price
        expected_payoff = np.maximum(test_strike_price - test_asset_grid, 0)

        assert np.array_equal(result[:,-1], expected_payoff)

    def test_check_valid_option_type(self):
        self.equation.option_type = 'woo'

        with pytest.raises(ValueError, match="Invalid option type - please choose between call/put"):
            solver.BlackScholesExplicitSolver(self.equation).solve().get_result()

    # crank-nicolson method tests

    def test_check_lower_boundary_for_call_cn(self):
        result = solver.BlackScholesCNSolver(self.equation).solve().get_result()
        assert np.all(result[0,:]) == 0

    def test_check_terminal_condition_for_call_cn(self):
        result = solver.BlackScholesCNSolver(self.equation).solve().get_result()

        test_asset_grid = self.equation.generate_asset_grid()
        test_strike_price = self.equation.strike_price
        expected_payoff = np.maximum(test_asset_grid - test_strike_price, 0)

        assert np.array_equal(result[:, -1], expected_payoff)

    def test_check_terminal_condition_for_put_cn(self):
        self.equation.option_type = 'put'
        result = solver.BlackScholesCNSolver(self.equation).solve().get_result()

        test_asset_grid = self.equation.generate_asset_grid()
        test_strike_price = self.equation.strike_price
        expected_payoff = np.maximum(test_strike_price - test_asset_grid, 0)

        assert np.array_equal(result[:,-1], expected_payoff)

    def test_check_absolute_difference_between_two_results(self):
        result1 = solver.BlackScholesExplicitSolver(self.equation).solve()
        result2 = solver.BlackScholesCNSolver(self.equation).solve()
        u1 = result1.get_result()
        u2 = result2.get_result()
        diff = u1 - u2

        assert np.max(np.abs(diff)) < 1e-2

    def test_convergence_between_interpolated_data(self):
        result1 = solver.BlackScholesExplicitSolver(self.equation).solve()
        result2 = solver.BlackScholesCNSolver(self.equation).solve()
        u1 = result1.get_result()
        u2 = result2.get_result()

        data1 = utility.RBFInterpolator(u1, 0.1, 0.03).interpolate(4,20)
        data2 = utility.RBFInterpolator(u2,0.1, 0.03).interpolate(4,20)

        diff = np.abs(data1 - data2)

        assert diff < 1e-4





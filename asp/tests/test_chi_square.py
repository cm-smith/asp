from unittest import TestCase
import numpy as np
from asp.chi_square import ChiSquare

sample_size_allowed_error = 5
power_allowed_error = 0.02
mde_allowed_error = 0.01

class TestChiSquare(TestCase):
    def test_correct_analytic_sample_size(self):
        true_value = 17610

        experiment = ChiSquare(
            sample_size=115
            ,test_splits=[0.5, 0.5]
            ,natural_rate=0.65
            ,absolute_effects=[0.02, 0.]
            ,alpha=0.05
            ,beta=0.2
        )
        asp_value = experiment.solve_sample_size()
        self.assertTrue(np.abs(asp_value-true_value) <= sample_size_allowed_error)

    def test_correct_analytic_power(self):
        true_value = 0.58

        experiment = ChiSquare(
            sample_size=140
            ,test_splits=[0.5, 0.5]
            ,natural_rate=0.43
            ,absolute_effects=[-0.15, 0.]
            ,alpha=0.10
            ,beta=0.2
        )
        asp_value = experiment.solve_power()
        self.assertTrue(np.abs(asp_value-true_value) <= power_allowed_error)

    def test_correct_analytic_mde(self):
        true_value = 0.084

        experiment = ChiSquare(
            sample_size=1146
            ,test_splits=[0.5, 0.5]
            ,natural_rate=0.65
            ,absolute_effects=[0.08, 0.]
            ,alpha=0.025
            ,beta=0.2
        )
        asp_value = experiment.solve_absolute_mde_test_group(0)
        self.assertTrue(np.abs(asp_value-true_value) <= mde_allowed_error)

from unittest import TestCase
import numpy as np
from asp.f_test import FTest

sample_size_allowed_error = 5
power_allowed_error = 0.01
mde_allowed_error = 0.02

class TestFTest(TestCase):
    def test_correct_analytic_sample_size(self):
        true_value = 132

        experiment = FTest(
            sample_size=10
            ,test_splits=[0.8333, 0.1667]
            ,mu=16.
            ,sigma=4.5
            ,absolute_effects=[-2., 0.]
            ,alpha=0.10
            ,beta=0.40
        )
        asp_value = experiment.solve_sample_size()
        self.assertTrue(np.abs(asp_value-true_value) <= sample_size_allowed_error)

    def test_correct_analytic_power(self):
        true_value = 0.72

        experiment = FTest(
            sample_size=237
            ,test_splits=[0.8333, 0.1667]
            ,mu=16.
            ,sigma=4.5
            ,absolute_effects=[-2., 0.]
            ,alpha=0.05
            ,beta=0.20
        )
        asp_value = experiment.solve_power()
        self.assertTrue(np.abs(asp_value-true_value) <= power_allowed_error)

    def test_correct_analytic_mde(self):
        true_value = 2.8

        experiment = FTest(
            sample_size=195
            ,test_splits=[0.8333, 0.1667]
            ,mu=16.
            ,sigma=4.5
            ,absolute_effects=[-1., 0.]
            ,alpha=0.05
            ,beta=0.10
        )
        asp_value = experiment.solve_absolute_mde_test_group(0)
        print("ASP value:", asp_value)
        print("True value:", -2.8)
        self.assertTrue(np.abs(asp_value-true_value) <= mde_allowed_error)

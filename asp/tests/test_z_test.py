from unittest import TestCase
import numpy as np
from asp.z_test import TwoProportionsZTest

sample_size_allowed_error = 15
power_allowed_error = 0.02
mde_allowed_error = 0.01
test_split_allowed_error = 0.01

class TestZTest(TestCase):
    def test_correct_analytic_sample_size(self):
        true_value = 172

        treat_n = 115
        power = 0.80
        alpha = 0.05
        control_mean = 0.65
        treat_mean = 0.85
        control_over_treat = 0.5

        sample_size = int(treat_n * control_over_treat + treat_n)
        test_split = 1./(1+control_over_treat)

        experiment = TwoProportionsZTest(
            sample_size=sample_size
            ,test_split=test_split
            ,natural_rate=control_mean
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
        )
        asp_value = experiment.solve_sample_size()
        self.assertTrue(np.abs(asp_value-true_value) <= sample_size_allowed_error)

    def test_correct_analytic_power(self):
        true_value = 0.7126

        treat_n = 70
        power = 0.0
        alpha = 0.10
        control_mean = 0.43
        treat_mean = 0.28
        control_over_treat = 2.0

        sample_size = int(treat_n * control_over_treat + treat_n)
        test_split = 1./(1+control_over_treat)

        experiment = TwoProportionsZTest(
            sample_size=sample_size
            ,test_split=test_split
            ,natural_rate=control_mean
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
        )
        asp_value = experiment.solve_power()
        self.assertTrue(np.abs(asp_value-true_value) <= power_allowed_error)

    def test_correct_analytic_mde(self):
        true_value = 0.08

        treat_n = 573
        power = 0.80
        alpha = 0.025
        control_mean = 0.65
        treat_mean = 0.73
        control_over_treat = 1.2

        sample_size = int(treat_n * control_over_treat + treat_n)
        test_split = 1./(1+control_over_treat)

        experiment = TwoProportionsZTest(
            sample_size=sample_size
            ,test_split=test_split
            ,natural_rate=control_mean
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
        )
        asp_value = experiment.solve_absolute_mde()
        self.assertTrue(np.abs(asp_value-true_value) <= mde_allowed_error)

    def test_correct_analytic_test_split(self):
        true_value = 0.45454545

        treat_n = 573
        power = 0.80
        alpha = 0.025
        control_mean = 0.65
        treat_mean = 0.73
        control_over_treat = 1.0

        sample_size = int(treat_n * control_over_treat + treat_n)
        test_split = 1./(1+control_over_treat)

        experiment = TwoProportionsZTest(
            sample_size=sample_size
            ,test_split=test_split
            ,natural_rate=control_mean
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
        )
        asp_value = experiment.solve_test_split()
        self.assertTrue(np.abs(asp_value-true_value) <= test_split_allowed_error)

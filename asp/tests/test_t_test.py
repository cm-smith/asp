from unittest import TestCase
import numpy as np
from asp.t_test import TTest

sample_size_allowed_error = 1
power_allowed_error = 0.01
mde_allowed_error = 0.01
test_split_allowed_error = 0.01

class TestTTest(TestCase):
    ###################
    # TWO-SIDED TESTS #
    ###################
    def test_correct_analytic_sample_size_two_sided(self):
        true_value = 132

        treat_n = 1.
        power = 0.6
        alpha = 0.1
        control_mean = 16
        treat_mean = 14
        sigma = 4.5
        control_over_treat = 0.2
        alternative = 'two-sided'

        sample_size = int(treat_n * control_over_treat + treat_n)
        test_split = 1./(1+control_over_treat)

        experiment = TTest(
            sample_size=sample_size
            ,test_split=test_split
            ,mu=control_mean
            ,sigma=sigma
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
            ,alternative=alternative
        )
        asp_value = experiment.solve_sample_size()
        self.assertTrue(np.abs(asp_value - true_value) <= sample_size_allowed_error)

    def test_correct_analytic_power_two_sided(self):
        true_value = 0.7244

        treat_n = 198
        power = 0.0
        alpha = 0.05
        control_mean = 16
        treat_mean = 14
        sigma = 4.5
        control_over_treat = 0.2
        alternative = 'two-sided'

        sample_size = int(treat_n * control_over_treat + treat_n)
        test_split = 1./(1 + control_over_treat)

        experiment = TTest(
            sample_size=sample_size
            ,test_split=test_split
            ,mu=control_mean
            ,sigma=sigma
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
            ,alternative=alternative
        )
        asp_value = experiment.solve_power()
        self.assertTrue(np.abs(asp_value-true_value) <= power_allowed_error)

    def test_correct_analytic_mde_two_sided(self):
        true_value = 2.81

        treat_n = 163
        power = 0.90
        alpha = 0.05
        control_mean = 16
        treat_mean = 17
        sigma = 4.5
        control_over_treat = 0.2
        alternative = 'two-sided'

        sample_size = int(treat_n * control_over_treat + treat_n)
        test_split = 1./(1 + control_over_treat)

        experiment = TTest(
            sample_size=sample_size
            ,test_split=test_split
            ,mu=control_mean
            ,sigma=sigma
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
            ,alternative=alternative
        )
        asp_value = experiment.solve_absolute_mde()
        self.assertTrue(np.abs(asp_value-true_value) <= mde_allowed_error)

    def test_correct_analytic_test_split_two_sided(self):
        true_value = 0.83333

        treat_n = 163
        power = 0.90
        alpha = 0.05
        control_mean = 16
        treat_mean = 18.81
        sigma = 4.5
        control_over_treat = 1.
        alternative = 'two-sided'

        sample_size = int(treat_n * control_over_treat + treat_n)
        test_split = 1./(1 + control_over_treat)

        experiment = TTest(
            sample_size=sample_size
            ,test_split=test_split
            ,mu=control_mean
            ,sigma=sigma
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
            ,alternative=alternative
        )
        asp_value = experiment.solve_test_split()
        self.assertTrue(np.abs(asp_value-true_value) <= test_split_allowed_error)

    ###################
    # ONE-SIDED TESTS #
    ###################
    def test_correct_analytic_sample_size_one_sided(self):
        true_value = 200

        control_n = 163
        power = 0.80
        alpha = 0.05
        control_mean = 132.86
        treat_mean = 127.44
        sigma = 15.34
        control_over_treat = 1.0
        alternative = 'one-sided'

        sample_size = 1.
        test_split = 1./(1+control_over_treat)

        experiment = TTest(
            sample_size=sample_size
            ,test_split=test_split
            ,mu=control_mean
            ,sigma=sigma
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
            ,alternative=alternative
        )
        asp_value = experiment.solve_sample_size()
        self.assertTrue(np.abs(asp_value - true_value) <= sample_size_allowed_error)

    def test_correct_analytic_power_one_sided(self):
        true_value = 0.3634

        control_n = 130
        power = 0.
        alpha = 0.05
        control_mean = 86.3
        treat_mean = 90.2
        sigma = 24.3
        control_over_treat = 1.
        alternative = 'one-sided'

        sample_size = int(control_n * (1./control_over_treat) + control_n)
        test_split = 1./(1 + control_over_treat)

        experiment = TTest(
            sample_size=sample_size
            ,test_split=test_split
            ,mu=control_mean
            ,sigma=sigma
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
            ,alternative=alternative
        )
        asp_value = experiment.solve_power()
        self.assertTrue(np.abs(asp_value-true_value) <= power_allowed_error)

    def test_correct_analytic_mde_one_sided(self):
        true_value = 6.5

        control_n = 130
        power = 0.80
        alpha = 0.05
        control_mean = 86.3
        treat_mean = 87.
        sigma = 24.3
        control_over_treat = 0.5
        alternative = 'one-sided'

        sample_size = int(control_n * (1./control_over_treat) + control_n)
        test_split = 1./(1 + control_over_treat)

        experiment = TTest(
            sample_size=sample_size
            ,test_split=test_split
            ,mu=control_mean
            ,sigma=sigma
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
            ,alternative=alternative
        )
        asp_value = experiment.solve_absolute_mde()
        self.assertTrue(np.abs(asp_value-true_value) <= mde_allowed_error)

    def test_correct_analytic_test_split_one_sided(self):
        true_value = 0.66666

        control_n = 130
        power = 0.80
        alpha = 0.05
        control_mean = 86.3
        treat_mean = 92.8
        sigma = 24.3
        control_over_treat = 0.5
        alternative = 'one-sided'

        sample_size = int(control_n * (1./control_over_treat) + control_n)
        test_split = 1./(1 + control_over_treat)

        experiment = TTest(
            sample_size=sample_size
            ,test_split=test_split
            ,mu=control_mean
            ,sigma=sigma
            ,absolute_effect=(treat_mean - control_mean)
            ,alpha=alpha
            ,beta=(1-power)
            ,alternative=alternative
        )
        asp_value = experiment.solve_test_split()
        self.assertTrue(np.abs(asp_value-true_value) <= test_split_allowed_error)

# Author(s): Michael Smith

import numpy as np
import pandas as pd
from statsmodels.stats.power import FTestPower
from scipy.stats import f_oneway

class FTest():
    """F-test applied to an experiment with multiple experimental groups and a continuous outcome variable.

    Parameters
    ----------
    sample_size : positive int
        Sample size for experimental population
    test_splits : list of floats (0, 1) that adds to one
        Percent of sample size belonging to each experimental group
    mu : float
        Mean for the control group
    sigma : float
        Standard deviation for the control group
        This test currently assumes that all groups have equal variance
    absolute_effects : list of floats
        Absolute effect sizes due to treatments (use zero for control group)
    alpha : float (0, 1)
        Allowable Type I error for hypothesis testing (no effect); default 0.05
    beta : float (0, 1)
        Allowable Type II error for hypothesis testing (no effect); default 0.20
        Note that (1 - beta) is the statistical power for the experiment
    verbose : boolean
        Flag for verbose output; default False

    Example Usage
    -------------
    # Initialize F-test experiment
    experiment = FTest(
        sample_size=1000
        ,test_splits=[0.5, 0.3, 0.2]
        ,mu=9.7
        ,sigma=1.3
        ,absolute_effects=[0.05, 0.02, 0.]
    )

    # Ignoring defined power (beta), solve for power given other experimental constraints
    experiment.solve_power()
    >>> 0.0685

    # Ignoring defined sample size, solve for sample size given other experimental constraints
    experiment.solve_sample_size()
    >>> 39813

    # Ignoring defined absolute effect size, solve for minimal (absolute) detectable effect given other experimental constraints
    # We linearly scale the original effect sizes until the system attains power
    experiment.solve_absolute_mde()
    >>> [0.315, 0.126, 0.]

    # Ignoring defined absolute effect size for a specific test group, solve for the MDE for that test cell
    experiment.solve_absolute_mde_test_group(test_cell=1)
    >>> 0.311

    # Simulate a single experiment to produce (F_stat, p_value, effect_size_point_estimates)
    experiment.simulate()
    >>> (1.032, 0.356, [-0.013, 0.002, 0.138])

    # Simulate many experiments to estimate % the correct-signed effect is detected
    experiment.estimate_correct_signed_effect()
    >>> [0.805, 0.606, 0.]
    """

    def __init__(self, sample_size, test_splits, mu, sigma, absolute_effects, alpha=0.05, beta=0.20, verbose=False):
        # User defined
        self.sample_size = sample_size
        self.test_splits = np.array(test_splits)
        self.mu = mu
        self.sigma = sigma
        self.absolute_effects = np.array(absolute_effects)

        # Check user-defined data
        assert self.sample_size > 0, \
            "F-test requires sample size greater than zero. " \
            "Received sample size (" + str(self.sample_size) + ")."

        assert round(self.test_splits.sum(), 6) == 1.0, \
            "F-test requires that you define how the sample size is distributed " \
            "across each experimental group.\n Sum of test splits must be one. " \
            "Received sum of " + str(self.test_splits.sum())

        assert len(test_splits) == len(absolute_effects), \
            "F-test requires equal array sizes for test splits and effect sizes. " \
            "Received inputs: " + str(self.test_splits) + " " + str(self.absolute_effects)

        # Default or defined
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.df_num = len(self.test_splits) - 1
        self.df_denom = self.sample_size - len(self.test_splits)

    def __repr__(self):
        return "FTest"

    def normalized_effect_size(self):
        """Apply Cohen's f formulation for normalized effect size.

        f = sqrt( sigma_m^2 ) / sigma,

        where `sigma` is the pooled standard deviation for the system and
        `sigma_m^2` is defined as:

        sigma_m^2 = Sum_i test_split_i * (mu_i - mu_pop)^2

        Returns
        -------
        f : float
            Cohen's f estimate given absolute effect sizes for experimental system
        """
        mus = self.mu + self.absolute_effects
        pop_mu = (mus * self.test_splits).sum()
        sigma2_m = (self.test_splits * np.square(mus - pop_mu)).sum()
        f = np.sqrt(sigma2_m) / self.sigma
        return f

    def solve_power(self):
        """Ignoring defined beta (Type II error), solve for statistical power given
        other defined experimental constraints. This method utilized the StatsModels
        functionality `solve_power`.

        Returns
        -------
        power : float (0, 1)
            Statistical power of experiment
        """
        e = self.normalized_effect_size()
        power = FTestPower().solve_power(
            effect_size=e
            ,df_num=self.df_denom
            ,df_denom=self.df_num
            ,alpha=self.alpha
            ,power=None
            ,ncc=1
        )
        return power

    def solve_sample_size(self):
        """Ignoring defined sample size, solves for sample size given other defined
        experimental constraints. This method utilizes the StatsModels functionality
        `solve_power`.

        Returns
        -------
        sample_size : int > 0
            Sample size required to run experiment with other constraints
        """
        e = self.normalized_effect_size()
        df_denom_solve = FTestPower().solve_power(
            effect_size=e
            ,df_num=None
            ,df_denom=self.df_num
            ,alpha=self.alpha
            ,power=(1 - self.beta)
            ,ncc=1
        )
        n = int(df_denom_solve + len(self.test_splits))
        return n

    def solve_absolute_mde(self):
        """Ignoring defined absolute effect size, solve for absolute Minimal
        Detectable Effect given other defined experimental constraints. To attain
        power, we scale ALL effect sizes linearly.

        Derivation for this method is available in "Examples > ANOVA Derivations"

        Returns
        -------
        min_absolute_effect : list of floats
            Absolute MDE for experiment
        """
        e = FTestPower().solve_power(
            effect_size=None
            ,df_num=self.df_denom
            ,df_denom=self.df_num
            ,alpha=self.alpha
            ,power=(1 - self.beta)
            ,ncc=1
        )

        Y = (self.test_splits * self.absolute_effects).sum()
        num1 = np.square(self.absolute_effects - Y)
        num = (self.test_splits * num1).sum()

        a = f * self.sigma / np.sqrt(num)
        return a * self.absolute_effects

    def solve_absolute_mde_test_group(self, test_cell):
        """Keeping all experimental constraints constant, solve for the absolute
        MDE (minimal detectable effect) for test group i.

        Derivation for this method is available in "Examples > ANOVA Derivations"

        Inputs
        ------
        test_cell : int
            Index of test cell to analyze; must be between 0 and (# Test Groups - 1)
            This correspond with the `test_spits` array in the class
        
        Returns
        -------
        min_absolute_effect : float
            Absolute MDE for test group of interest
        """
        assert test_cell >= 0 and test_cell < len(self.test_splits), \
            "Test to analyze muse be in range (0,"+str(len(self.test_splits))+")"

        f = FTestPower().solve_power(
            effect_size=None
            ,df_num=self.df_denom
            ,df_denom=self.df_num
            ,alpha=self.alpha
            ,power=(1 - self.beta)
            ,ncc=1
        )

        # Fix test group of interest
        tj = self.test_splits[test_cell]
        ej = self.absolute_effects[test_cell]

        f2_sigma2 = np.square(f) * np.square(self.sigma)
        X = (self.test_splits * self.absolute_effects).sum() - (tj*ej)
        X_star = (self.test_splits * np.square(self.absolute_effects)).sum() - (tj*ej*ej)

        a = tj*(1-tj)
        b = -2*tj*X
        c = X_star - X*X - f2_sigma2

        roots = np.roots([a, b, c])

        if np.iscomplex(roots).sum() > 0:
            print("MDE calculation found complex roots; returning " \
                "closest effect size to desired power (i.e., the real part of " \
                "the complex roots) for test group " + str(test_cell))
            min_absolute_effect = roots[0].real
        elif tj >= 0: min_absolute_effect = np.max(roots)
        else: min_absolute_effect = np.min(roots)

        return min_absolute_effect

    def simulate(self):
        """This simulation assumes that we are testing for an `effect` in a single
        experiment.

        Returns
        -------
        f_stat : float
            The F statistic from F-test in SciPy
        p_value : float [0, 1]
            The p-value from F-test in SciPy
        effect_point_estimates : float
            The effect size point estimates observed in the treatment groups
        """
        observations = [
            np.random.normal(
                self.mu + absolute_effect
                ,self.sigma
                ,int(self.sample_size * self.test_splits[i])
            )
            for i, absolute_effect in enumerate(self.absolute_effects)
        ]

        effect_point_estimates = [
            round(test_observations.mean()-self.mu, 4)
            for test_observations in observations
        ]
        f_stat, p_value = f_oneway(*observations)
        return f_stat, p_value, effect_point_estimates

    def batch_simulation(self, iters=10000):
        """Run a batch simulation to estimate experimental outcomes

        Returns
        -------
        power : float (0, 1)
            Statistical power estimated from batch simulation
        pct_correct_sign : float (0, 1)
            Percent of the time the effect observed was "correct signed" in the batch simulation
        """
        power_cnt = 0
        correct_sign_cnt = 0

        for i in range(iters):
            if (self.verbose) and (i>0) and (i % (iters/10) == 0):
                print(i, " / ", iters)
            f_stat, p_value, effect_point_estimates = self.simulate()
            power_cnt += (p_value < self.alpha)
            correct_sign_cnt += (effect_point_estimates * self.absolute_effect > 0)

        if self.verbose: print(iters, " / ", iters)
        power = round(power_cnt / float(iters), 5)
        pct_correct_sign = round(correct_sign_cnt / float(iters), 5)
        return power, pct_correct_sign

    def estimate_correct_signed_effect(self, iters=None):
        """Estimate percent of the time we observe the correct signed effect using
        the defined experimental parameters.

        Returns
        -------
        pct_correct_sign : float (0, 1)
            Percent correct sign was observed in batch simulation estimation
        """
        if iters is None:
            _, pct_correct_sign = self.batch_simulation()
        else:
            _, pct_correct_sign = self.batch_simulation(iters=iters)

        return pct_correct_sign


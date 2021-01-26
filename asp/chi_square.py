# Author(s): Michael Smith

import numpy as np
import pandas as pd
from statsmodels.stats.power import GofChisquarePower
from statsmodels.stats.proportion import proportions_chisquare

class ChiSquare():
    """Chi square statistic test applied to an experiment with multiple treatment
    groups and a binary categorical outcome variable.

    Parameters
    ----------
    sample_size : positive int
        Sample size for experimental population
    test_splits : list of floats (0, 1) that adds to one
        Defines how the sample size is split across experimental groups
    natural_rate : float (0, 1)
        Natural rate for success (1) of binary response variable (0/1)
    absolute_effects : list of floats (0, 1)
        Absolute effect sizes that we desire to detect in each experimental group
    alpha : float (0, 1)
        Allowable Type I error for hypothesis testing (no effect); default 0.05
    beta : float (0, 1)
        Allowable Type II error for hypothesis testing (no effect); default 0.20
        Note that (1 - beta) is the statistical power for the experiment
    verbose : boolean
        Flag for verbose output; default False

    Example Usage
    -------------
    # Initialize chi square experiment
    experiment = ChiSquare(
        sample_size=1000
        ,test_splits=[0.4,0.4,0.2]
        ,natural_rate=0.20
        ,absolute_effects=[0.05,0.04,0.0]
    )

    # Ignoring defined power (beta), solve for power given other experimental constraints
    experiment.solve_power()
    >>> 0.3145

    # Ignoring defined sample size, solve for sample size given other experimental constraints
    experiment.solve_sample_size()
    >>> 3292

    # Ignoring defined absolute effect size, solve for minimal (absolute) detectable effect given other experimental constraints
    # We linearly scale the original effect sizes until the system attains power
    experiment.solve_absolute_mde()
    >>> [0.0998, 0.0798, 0.]

    # Ignoring defined absolute effect size for a specific test group, solve for the MDE for that test cell
    experiment.solve_absolute_mde_test_group(test_cell=1)
    >>> 0.0963

    # Simulate a single experiment to produce (chi2_stat, p_value, effect_size_point_estimates)
    experiment.simulate()
    >>> (1.716, 0.423, [0.0275, 0.0375, 0.])

    # Simulate many experiments to estimate % the correct-signed effect is detected
    experiment.estimate_correct_signed_effect()
    >>> [0.9975, 0.992, 0.]
    """

    def __init__(self, sample_size, test_splits, natural_rate, absolute_effects, alpha=0.05, beta=0.20, verbose=False):
        # User defined
        self.sample_size = sample_size
        self.test_splits = np.array(test_splits)
        self.natural_rate = natural_rate
        self.absolute_effects = np.array(absolute_effects)

        # Check user-defined data
        assert self.sample_size > 0, \
            "Chi Square test requires sample size greater than zero. " \
            "Received sample size (" + str(self.sample_size) + ")."

        assert round(self.test_splits.sum(),4) == 1, \
            "Chi Square test requires a list of ratios that specify how the " \
            "total sample size is distributed across experimental groups. This " \
            "list must add to one.\nReceived input: " + str(self.test_splits)

        assert len(self.test_splits) == len(self.absolute_effects), \
            "Chi Square test requires a list of absolute effect sizes " \
            "and a list of test splits of the same size.\nReceived input: " + \
            str(self.absolute_effects) + " " + str(self.test_splits)

        assert np.min(self.natural_rate + self.absolute_effects) > 0 \
            and np.max(self.natural_rate + self.absolute_effects) < 1, \
            "Chi Square test cannot detect given effect size with rate greater " \
            "than 1 or less than zero. Received absolute effect sizes: " + \
            str(self.absolute_effects)

        # Default or defined
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.n_groups = len(self.test_splits)

    def __repr__(self):
        return "ChiSquare"

    def normalized_effect_size(self):
        """Solve for Cohen's w normalized effect size. This is derived from
        a contingency table for the experiment with i-many cells:

        w = sqrt( sum_i( (E_i - O_i)^2 / E_i) ),

        where E_i is the probability of cell i under the Null Hypothesis (i.e.,
        the "expected" value given no effect) and O_i is the probability of cell i
        under the Alternative Hypothesis (i.e., the "observed" value).

        Returns
        -------
        w : float
            Cohen's w estimate given absolute effect sizes
        """
        observed_success = self.test_splits * (self.absolute_effects + self.natural_rate)
        observed_failure = self.test_splits * (1 - self.absolute_effects - self.natural_rate)
        observed_table = np.stack([observed_success, observed_failure])

        # Note that marginal success is `observed_table[0].sum()`
        expected_success = observed_table[0].sum() * self.test_splits
        expected_failure = observed_table[1].sum() * self.test_splits
        expected_table = np.stack([expected_success, expected_failure])
        
        w2 = np.sum(np.square((expected_table-observed_table))/expected_table)
        return np.sqrt(w2)

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
        power = GofChisquarePower().solve_power(
            effect_size=e
            ,nobs=self.sample_size
            ,alpha=self.alpha
            ,n_bins=self.n_groups
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
        sample_size = GofChisquarePower().solve_power(
            effect_size=e
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,n_bins=self.n_groups
        )
        return int(np.ceil(sample_size))

    def solve_absolute_mde(self):
        """Ignoring defined absolute effect size, solve for absolute Minimal
        Detectable Effect given other defined experimental constraints. Solve
        such that all absolute effects are scaled linearly to reach sufficient power.

        Derivation for this method is available in "Examples > ANOVA Derivations"

        Returns
        -------
        min_absolute_effect : list of floats (0, 1)
            Absolute MDE for experiment
        """
        e = NormalIndPower().solve_power(
            nobs=self.sample_size
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,n_bins=self.n_groups
        )

        w2 = np.power(e, 2)
        p = self.natural_rate

        Y = (self.test_splits * self.absolute_effects).sum()
        num1 = np.square(Y - self.absolute_effects)
        num = (self.test_splits * num1).sum()

        a = num + w2*Y*Y
        b = 2*w2*p*Y - w2*Y
        c = w2*p*p - w2*p

        roots = np.roots([a, b, c])

        if self.absolute_effects.sum() > 0:
            root = roots.max()
        else:
            root = roots.min()

        return root*self.absolute_effects

        return min_absolute_effect

    def solve_absolute_mde_test_group(self, test_cell):
        """Keeping all experimental constraints constant, solve for the absolute
        MDE (minimal detectable effect) for test group i.

        Derivation for this method is available in "Examples > ANOVA Derivations"

        Inputs
        ------
        test_cell : int
            Index of test cell to analyze; must be between 0 and (# Test Groups - 1)
            Corresponds with `test_splits` and `absolute_effects` in class

        Returns
        -------
        min_absolute_effect : float
            Absolute MDE for test group of interest
        """
        assert test_cell >= 0 and test_cell < len(self.test_splits), \
            "Test cell to analyze must be in range (0,"+str(len(self.test_splits))+")"

        e = GofChisquarePower().solve_power(
            nobs=self.sample_size
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,n_bins=self.n_groups
        )
        w2 = np.power(e, 2)

        e_j = self.absolute_effects[test_cell]
        t_j = self.test_splits[test_cell]
        p = self.natural_rate

        X = np.sum(self.test_splits * self.absolute_effects) - e_j*t_j
        other_const = np.sum(self.test_splits * np.square(self.absolute_effects)) - t_j*e_j*e_j

        a = t_j*t_j*(w2-1)+t_j
        b = w2*t_j*(2*p-1)+2*X*t_j*(w2-1)
        c = w2*p*(p-1)+2*w2*X*p-w2*X*(1-X)-X*X+other_const

        roots = np.roots([a, b, c])

        if np.iscomplex(roots).sum() > 0:
            print("MDE calculation found complex roots; returning " \
                "closest effect size to desired power (i.e., the real part of " \
                "the complex roots) for test group " + str(test_cell))
            min_absolute_effect = roots[0].real
        elif t_j >= 0: min_absolute_effect = np.max(roots)
        else: min_absolute_effect = np.min(roots)

        return min_absolute_effect

    def simulate(self):
        """This simulation assumes that we are testing for an `effect` in a single
        experiment.

        Returns
        -------
        chi2 : float
            The chi2 statistic from chi2 test in StatsModels
        p_value : float [0, 1]
            The p-value from chi2 test in StatsModels
        effect_point_estimates : list of float
            The effect size point estimates observed in the treatment groups
        """
        observations = [
            np.random.binomial(
                1
                ,self.natural_rate + absolute_effeect
                ,int(self.sample_size * self.test_splits[i])
            ).sum()
            for i, absolute_effect in enumerate(self.absolute_effects)
        ]

        effect_point_estimates = round(
            np.array(observations)/(self.test_splits*self.sample_size) - self.natural_rate
            ,4
        )
        chi2, p_value, _ = proportions_chisquare(
            observations
            ,(self.test_splits * self.sample_size).astype(int)
        )
        return chi2, p_value, effect_point_estimates

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
            chi2, p_value, effect_point_estimates = self.simulate()
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


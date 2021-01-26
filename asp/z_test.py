# Author(s): Michael Smith

"""NOTES:
The following relationship exists between StatsModels functions for Two Proportions
z Test and Chi Square Test:

```python
power1 = GofChisquarePower().solve_power(cohens_w, nobs=n, alpha=alpha, n_bins=2)
power2 = NormalIndPower().solve_power(cohens_h, nobs1=n_treat, alpha=alpha)

# Power 1 approximates Power 2
print("Power 1:", power1)
print("Power 2:", power2)
```
"""

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

class TwoProportionsZTest():
    """z-test applied to an experiment with two experimental groups--i.e., treatment
    and hold-out--and a binary outcome variable (0/1).

    Parameters
    ----------
    sample_size : positive int
        Sample size for experimental population
    test_split : float (0, 1)
        Percent of sample size belonging in treatment group
    natural_rate : float (0, 1)
        Base rate for success of binary response variable (proportion in control group)
    absolute_effect : float
        Absolute effect size due to treatment
    alpha : float (0, 1)
        Allowable Type I error for hypothesis testing (no effect); default 0.05
    beta : float (0, 1)
        Allowable Type II error for hypothesis testing (no effect); default 0.20
        Note that (1 - beta) is the statistical power for the experiment
    verbose : boolean
        Flag for verbose output; default False

    Example Usage
    -------------
    # Initialize two proportions z-test experiment
    experiment = TwoProportionsZTest(
        sample_size=1000
        ,test_split=0.50
        ,natural_rate=0.20
        ,absolute_effect=0.05
    )

    # Ignoring defined power (beta), solve for power given other experimental constraints
    experiment.solve_power()
    >>> 0.474

    # Ignoring defined sample size, solve for sample size given other experimental constraints
    experiment.solve_sample_size()
    >>> 2184

    # Ignoring defined absolute effect size, solve for minimal (absolute) detectable effect given other experimental constraints
    experiment.solve_absolute_mde()
    >>> 0.0752

    # Ignoring defined test split, solve for required test_split given other experimental constraints
    experiment.solve_test_split()
    >>> (Convergence warning, since sample size does not adequately power experiment right now)

    # Simulate a single experiment to produce (z_stat, p_value, effect_size_point_estimate)
    experiment.simulate()
    >>> (2.305, 0.021, 0.062)

    # Simulate many experiments to estimate % the correct-signed effect is detected
    experiment.estimate_correct_signed_effect()
    >>> 0.9955
    """

    def __init__(self, sample_size, test_split, natural_rate, absolute_effect, alpha=0.05, beta=0.20, verbose=False):
        # User defined
        self.sample_size = sample_size
        self.test_split = test_split
        self.natural_rate = natural_rate
        self.absolute_effect = absolute_effect

        # Check user-defined data
        assert self.sample_size > 0, \
            "z-test requires sample size greater than zero. " \
            "Received sample size (" + str(self.sample_size) + ")."

        assert self.test_split > 0 and self.test_split < 1, \
            "z-test requires test-split between zero and one. " \
            "Received test split (" + str(self.test_split) + ")."

        assert self.natural_rate > 0 and self.natural_rate < 1, \
            "z-test requires a control group rate between zero and one. " \
            "Received natural rate (" + str(self.natural_rate) + ")."

        assert self.natural_rate + self.absolute_effect > 0 \
            and self.natural_rate + self.absolute_effect < 1, \
            "z-test cannot detect given effect size. Received absolute effect " \
            "size (" + str(self.absolute_effect) + ") and natural rate (" + \
            str(self.natural_rate) + ")."

        # Default or defined
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

    def __repr__(self):
        return "TwoProportionsZTest"

    def normalized_effect_size(self):
        """Apply Cohen's h formulation for normalized effect size.

        h = 2 * [arcsin( sqrt(p1) ) - arcsin( sqrt(p2) )],

        where p1 and p2 are the rates of success for the two experimental groups.

        Returns
        -------
        h : float
            Cohen's h estimate given absolute effect size for treatment group
        """
        #h = proportion_effectsize(self.natural_rate + self.absolute_effect, self.natural_rate)
        h1 = np.arcsin(np.sqrt(self.natural_rate + self.absolute_effect))
        h2 = np.arcsin(np.sqrt(self.natural_rate))
        h = 2 * (h1 - h2)
        return h

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
        power = NormalIndPower().solve_power(
            effect_size=e
            ,nobs1=np.ceil(self.sample_size * self.test_split)
            ,alpha=self.alpha
            ,ratio=(1 - self.test_split) / self.test_split
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
        n_treat = NormalIndPower().solve_power(
            effect_size=e
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,ratio=(1 - self.test_split) / self.test_split
        )
        return int(np.ceil(n_treat / self.test_split))

    def solve_absolute_mde(self):
        """Ignoring defined absolute effect size, solve for absolute Minimal
        Detectable Effect given other defined experimental constraints. We leverage
        the StatsModels functionality, but need to solve for the "absolute" MDE. The
        StatsModel function returns the normalized Cohen's d effect size:

        MDE_Absolute = sin^2(arcsin(sqrt(natural_rate)) + cohens_h/2) - natural_rate

        Returns
        -------
        min_absolute_effect : float (0, 1)
            Absolute MDE for experiment
        """
        e = NormalIndPower().solve_power(
            nobs1=np.ceil(self.sample_size * self.test_split)
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,ratio=(1 - self.test_split) / self.test_split
        )
        sqrt_absolute_effect = np.sin(np.arcsin(np.sqrt(self.natural_rate))+e/2.)
        min_absolute_effect = np.power(sqrt_absolute_effect,2)-self.natural_rate
        return min_absolute_effect

    def solve_test_split(self):
        """Ignoring defined test split, solve for maximum test split.

        Returns
        -------
        max_test_split : float (0, 1)
            Maximum test split for experiment
        """
        e = self.normalized_effect_size()
        ratio = NormalIndPower().solve_power(
            effect_size=e
            ,nobs1=np.ceil(self.sample_size * self.test_split)
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,ratio=None
        )
        max_test_split = 1./(ratio + 1)
        return max_test_split

    def simulate(self):
        """This simulation assumes that we are testing for an `effect` in a single
        experiment.

        Returns
        -------
        z_stat : float
            The z statistic from z-test in StatsModels
        p_value : float [0, 1]
            The p-value from z-test in StatsModels
        effect_point_estimate : float
            The effect size point estimate observed in the treatment group
        """
        n_treat = int(np.ceil(self.sample_size * self.test_split))
        n_control = int(self.sample_size - n_treat)

        # Treatment
        exp_observations = [np.random.binomial(1, (self.natural_rate + self.absolute_effect), n_treat).sum()]
        # Control
        exp_observations.append(np.random.binomial(1, (self.natural_rate), n_control).sum())

        effect_point_estimate = round(
            exp_observations[0]/float(n_treat) - exp_observations[1]/float(n_control)
            ,4
        )
        z_stat, p_value = proportions_ztest(exp_observations, [n_treat, n_control])
        return z_stat, p_value, effect_point_estimate

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
            z_stat, p_value, effect_point_estimate = self.simulate()
            power_cnt += (p_value < self.alpha)
            correct_sign_cnt += (effect_point_estimate * self.absolute_effect > 0)

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


# Author(s): Michael Smith

import numpy as np
import pandas as pd
from statsmodels.stats.power import tt_ind_solve_power
from scipy.stats import ttest_ind

class TTest():
    """t-test applied to an experiment with two experimental groups--i.e., treatment
    and hold-out--and a continuous outcome variable.

    Parameters
    ----------
    sample_size : positive int
        Sample size for experimental population
    test_split : float (0, 1)
        Percent of sample size belonging in treatment group
    mu : float
        Mean for the control group
    sigma : float
        Standard deviation for the control group; currently assumes test group will
        have the same standard deviation
    absolute_effect : float
        Absolute effect size due to treatment
    alternative : string ('two-sided' or 'one-sided')
        Testing method; default is 'two-sided'
    alpha : float (0, 1)
        Allowable Type I error for hypothesis testing (no effect); default 0.05
    beta : float (0, 1)
        Allowable Type II error for hypothesis testing (no effect); default 0.20
        Note that (1 - beta) is the statistical power for the experiment
    verbose : boolean
        Flag for verbose output; default False

    Example Usage
    -------------
    # Initialize t-test experiment
    experiment = TTest(
        sample_size=1000
        ,test_split=0.50
        ,mu=9.7
        ,sigma=1.3
        ,absolute_effect=0.05
    )

    # Ignoring defined power (beta), solve for power given other experimental constraints
    experiment.solve_power()
    >>> 0.0932

    # Ignoring defined sample size, solve for sample size given other experimental constraints
    experiment.solve_sample_size()
    >>> 21226

    # Ignoring defined absolute effect size, solve for minimal (absolute) detectable effect given other experimental constraints
    experiment.solve_absolute_mde()
    >>> 0.2305

    # Ignoring defined test split, solve for required test_split given other experimental constraints
    experiment.solve_test_split()
    >>> (Convergence warning, since sample size does not adequately power experiment right now)

    # Simulate a single experiment to produce (t_stat, p_value, effect_size_point_estimate)
    experiment.simulate()
    >>> (1.938, 0.0529, 0.1613)

    # Simulate many experiments to estimate % the correct-signed effect is detected
    experiment.estimate_correct_signed_effect()
    >>> 0.7289
    """

    def __init__(self, sample_size, test_split, mu, sigma, absolute_effect, alternative='two-sided', alpha=0.05, beta=0.20, verbose=False):
        # User defined
        self.sample_size = sample_size
        self.test_split = test_split
        self.mu = mu
        self.sigma = sigma
        self.absolute_effect = absolute_effect

        # Check user-defined data
        assert self.sample_size > 0, \
            "t-test requires sample size greater than zero. " \
            "Received sample size (" + str(self.sample_size) + ")."

        assert self.test_split > 0 and self.test_split < 1, \
            "t-test requires test-split between zero and one. " \
            "Received test split (" + str(self.test_split) + ")."

        assert alternative in ['two-sided', 'one-sided'], \
            "t-test can either be 'two-sided' or 'one-sided'. " \
            "Received alternative (" + str(alternative) + ")."

        # Default or defined
        self.alternative = alternative
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

    def __repr__(self):
        return "TTest"

    def _alternative_direction(self):
        if self.alternative == 'two-sided': return self.alternative
        if self.absolute_effect < 0: return 'smaller'
        return 'larger'

    def normalized_effect_size(self):
        """Apply Cohen's d formulation for normalized effect size.

        d = absolute_effect_size / standard_deviation,

        where `standard_deviation` assumes that the two test groups will have equal
        variance.

        Returns
        -------
        d : float
            Cohen's d estimate given absolute effect size for treatment group
        """
        d = self.absolute_effect / self.sigma
        return d

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
        test_type = self._alternative_direction()
        power = tt_ind_solve_power(
            effect_size=e
            ,nobs1=np.ceil(self.sample_size * self.test_split)
            ,alpha=self.alpha
            ,ratio=(1 - self.test_split) / self.test_split
            ,alternative=test_type
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
        test_type = self._alternative_direction()
        n_treat = tt_ind_solve_power(
            effect_size=e
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,ratio=(1 - self.test_split) / self.test_split
            ,alternative=test_type
        )
        return int(np.ceil(n_treat / self.test_split))

    def solve_absolute_mde(self):
        """Ignoring defined absolute effect size, solve for absolute Minimal
        Detectable Effect given other defined experimental constraints. We leverage
        the StatsModels functionality, but need to solve for the "absolute" MDE. The
        StatsModel function returns the normalized Cohen's d effect size:

        d = absolute_effect_size / standard_deviation,

        Therefore, we solve for d * standard_deviation.

        Returns
        -------
        min_absolute_effect : float (0, 1)
            Absolute MDE for experiment
        """
        test_type = self._alternative_direction()
        e = tt_ind_solve_power(
            nobs1=np.ceil(self.sample_size * self.test_split)
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,ratio=(1 - self.test_split) / self.test_split
            ,alternative=test_type
        )
        min_absolute_effect = e * self.sigma

        if self.absolute_effect < 0 and min_absolute_effect > 0:
            min_absolute_effect *= -1

        return min_absolute_effect

    def solve_test_split(self):
        """Ignoring defined test split, solve for maximum test split.

        Returns
        -------
        max_test_split : float (0, 1)
            Maximum test split for experiment
        """
        e = self.normalized_effect_size()
        test_type = self._alternative_direction()
        ratio = tt_ind_solve_power(
            effect_size=e
            ,nobs1=np.ceil(self.sample_size * self.test_split)
            ,power=(1 - self.beta)
            ,alpha=self.alpha
            ,ratio=None
            ,alternative=test_type
        )
        max_test_split = 1./(ratio + 1)
        return max_test_split

    def simulate(self):
        """This simulation assumes that we are testing for an `effect` in a single
        experiment.

        Returns
        -------
        t_stat : float
            The t statistic from t-test in SciPy
        p_value : float [0, 1]
            The p-value from t-test in SciPy
        effect_point_estimate : float
            The effect size point estimate observed in the treatment group
        """
        n_treat = int(np.ceil(self.sample_size * self.test_split))
        n_control = int(self.sample_size - n_treat)

        treatment = np.random.normal(self.mu + self.absolute_effect, self.sigma, n_treat)
        control = np.random.normal(self.mu, self.sigma, n_control)

        effect_point_estimate = round(treatment.mean() - control.mean(), 4)
        t_stat, p_value = ttest_ind(treatment, control)
        return t_stat, p_value, effect_point_estimate

    def batch_simulation(self, iters=10000):
        """Run a batch simulation to estimate experimental outcomes

        Returns
        -------
        power : float (0, 1)
            Statistical power estimated from batch simulation
        pct_correct_sign : float (0, 1)
            Percent of the time the effect observed was "correct signed" in the batch simulation
        """
        test_type = self._alternative_direction()
        if test_type == 'larger':
            p_val_scale = 0.5
            t_stat_scale = 1.0
        elif test_type == 'smaller':
            p_val_scale = 0.5
            t_stat_scale = -1.0
        else:
            p_val_scale = 1.
            t_stat_scale = 0.

        power_cnt = 0
        correct_sign_cnt = 0

        for i in range(iters):
            if (self.verbose) and (i>0) and (i % (iters/10) == 0):
                print(i, " / ", iters)
            t_stat, p_value, effect_point_estimate = self.simulate()

            # https://stackoverflow.com/a/15984310
            # Reject the null hypothesis when...
            # (a) greater-than test --> p/2 < alpha and t > 0, and
            # (b) less-than test --> p/2 < alpha and t < 0
            power_cnt += (p_value * p_val_scale < self.alpha) * (t_stat * t_stat_scale >= 0)
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


# Assistant Stats Package (ASP)

There is a noticeable lack in Python's foundational statistics packages. The
go-to for analytics in Python has been [StatsModels](https://www.statsmodels.org/stable/index.html). This library aims to translate many foundational stats functions in R to a Python interface. However, the sparse documentation leads to a steep learning cureve, and many function lack flexibility. This repo is an attempt to improve the usage of StatsModels, as well as expanding the functionality. For instance, a goal of each statistical test is to include a simulation method, which can provide estimates for power, superiority/inferiority, etc.

# Available functionality

Currently, ASP supports the following experimental designs:

Design | ASP Class
--- | ---
Comparing two proportions | `TwoProportionsZTest`
Comparing two continuous metrics | `TTest` (one- or two-sided)
Comparing multiple proportions | `ChiSquare`
Comparing multiple continuous metrics | `FTest`

## Experimental Design Usage

This package is primarily intended to streamline the experimental design
process. Prior to running an A/B test, there may be questions about the
statistical power or minimal detectable effect. If the outcome is binary and we
are comparing two groups, we can apply `TwoProportionsZTest`:

```python
# Initialize z-test experiment
experiment = TwoProportionsZTest(
	sample_size=1000
	,test_split=0.50
	,natural_rate=0.20
	,absolute_effect=0.05
)

# Ignoring defined power (1-beta), solve for power given other experimental constraints
experiment.solve_power()
>>> 0.474

# Ignoring defined sample size, solve for sample size given other experimental constraints
experiment.solve_sample_size()
>>> 2184

# Ignoring defined absolute effect size, solve for minimal (absolute) detectable effect given other experimental constraints
experiment.solve_absolute_mde()
>>> 0.0752

# Ignoring defined test split, solve for required test split given other experimental constraints
experiment.solve_test_split()
>>> (Convergence warning, since sample size does not adequately power us right now)

# Simulate a single experiment to produce (z_stat, p_value, effect_size_point_estimate)
experiment.simulate()
>>> (2.305, 0.021, 0.062)

# Simulate many experiments to estimate percent of time the observed effect has the correct sign (+/-)
experiment.estimate_correct_signed_effect()
>>> 0.9955
```


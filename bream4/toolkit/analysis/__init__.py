"""
Analyses
========

An analysis is a call that can be performed outside an acquisition period.
It both takes and returns either data or preprocessed data.

Analyses are mostly specific to a certain procedure.
For example, there will be specific calls for analysis of data for the
dry electrode assessment.

There are two basic types of analyses. Those generating metrics from data,
and those that generate assessments from metrics.

Analyses generating metrics take data and a collection of functions as arguments.
Analyses generating assessments take metrics and passing criteria for those metrics and performs additional operations.

"""

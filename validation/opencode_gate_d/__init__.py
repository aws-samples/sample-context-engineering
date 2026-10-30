"""Gate D for the OpenCode port: the contract suite, the evidence contract and the evaluator.

Practice D (context graph) is request-only on OpenCode: it leaves messages out of one model request and appends one summary
block, and it never writes to the session. This package checks that claim, scenario by scenario, over recorded real sessions
and synthetic ones, and turns the result into a decision. It is a different gate from Gate A and never reclassifies it.
"""

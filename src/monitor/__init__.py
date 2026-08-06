"""The monitoring plane: prediction log, drift detectors, retraining trigger.

Only ``logger.py`` exists so far - it ships alongside the serving app in Week 3 (see
``PROJECT_PLAN.md`` §6/§8) so the drift detectors have real accumulated data the day
they're written, rather than starting from an empty log.
"""

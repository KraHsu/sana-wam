"""LIBERO benchmark client for sana-wam.

The simulator is intentionally kept outside the main sana-wam Python
environment.  Importing this package must therefore not import ``libero``,
``robosuite``, or MuJoCo eagerly.
"""

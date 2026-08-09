"""Compatibility import for the v0.6.3 robust production MPC.

The robust best-so-far / better-than-hold logic now lives in the single canonical
:mod:`rtc.tfv_mpc` implementation. This module remains only so intermediate v0.6.3 code and
external imports do not break.
"""

from .tfv_mpc import ContinuousTFVFirstMPC, TFVFirstMPCResult

__all__ = ["ContinuousTFVFirstMPC", "TFVFirstMPCResult"]

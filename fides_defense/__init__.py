# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""FIDES as an AgentDojo / AgentDyn prompt-injection defense baseline.

This package wraps the information-flow-control (IFC) planning loop from the
accompanying tutorial (``Tutorial.ipynb``) so it can be run as a defense over the
AgentDojo benchmark (and the AgentDyn suites: shopping / github / dailylife) with
the same command-line interface as the sibling baselines (CaMeL, DRIFT):

    python main.py MODEL --run-attack --attack important_instructions \
        --suite banking slack travel workspace --defense fides

The IFC core (lattices, labeled values, dynamic taint-tracking loop, policies) is a
faithful port of the tutorial -- it is NOT re-designed or "improved". Only a thin
adapter binds it to AgentDojo's runtime, tools and scoring harness.
"""

from fides_defense.lattice import (
    ConfidentialityLabel,
    IntegrityLabel,
    InverseLattice,
    Lattice,
    PowersetLattice,
    ProductLabel,
)
from fides_defense.core import (
    Action,
    PolicyViolation,
    Query,
    Response,
    ToolCall,
)

__all__ = [
    "ConfidentialityLabel",
    "IntegrityLabel",
    "InverseLattice",
    "Lattice",
    "PowersetLattice",
    "ProductLabel",
    "Action",
    "Query",
    "ToolCall",
    "Response",
    "PolicyViolation",
]

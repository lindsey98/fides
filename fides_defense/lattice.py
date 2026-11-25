# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Information-flow lattices.

A verbatim port of the lattice classes from ``Tutorial.ipynb`` (the "Information-Flow
Lattices" section). We define bounded join semi-lattices with a ``join`` operation, the
standard confidentiality lattice {Low, High}, the standard integrity lattice
{Trusted, Untrusted}, a powerset lattice ordered by subset inclusion, and the generic
``ProductLabel`` / ``InverseLattice`` combinators.

The product of the confidentiality and integrity lattices is used to label values;
``InverseLattice(PowersetLattice(readers))`` gives a confidentiality lattice ordered by
*readers* (a value may flow only to a context whose reader set is a subset).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, FrozenSet, Generic, TypeVar


class Lattice(ABC):
    """Abstract base class for (bounded) IFC lattices."""

    @abstractmethod
    def leq(self, other: Any) -> bool:
        """Returns True if self <= other in the lattice."""

    @abstractmethod
    def join(self, other: Any) -> Any:
        """Returns the least upper bound of self and other."""

    @abstractmethod
    def meet(self, other: Any) -> Any:
        """Returns the greatest lower bound of self and other."""

    @abstractmethod
    def __repr__(self) -> str: ...

    # --- Syntax sugar ---
    def __le__(self, other: "Lattice") -> bool:
        return self.leq(other)


class ConfidentialityLabel(Lattice):
    """Standard confidentiality lattice: LOW <= HIGH."""

    class Level(Enum):
        LOW = 0
        HIGH = 1

    def __init__(self, level: "ConfidentialityLabel.Level"):
        self.level = level

    def leq(self, other: "ConfidentialityLabel") -> bool:
        return self.level.value <= other.level.value

    def join(self, other: "ConfidentialityLabel") -> "ConfidentialityLabel":
        return other if self.leq(other) else self

    def meet(self, other: "ConfidentialityLabel") -> "ConfidentialityLabel":
        return self if self.leq(other) else other

    def __repr__(self):
        return f"{self.level.name}"

    @classmethod
    def low(cls):
        return cls(cls.Level.LOW)

    @classmethod
    def high(cls):
        return cls(cls.Level.HIGH)


class IntegrityLabel(Lattice):
    """Standard integrity lattice: TRUSTED <= UNTRUSTED."""

    class Level(Enum):
        TRUSTED = 0
        UNTRUSTED = 1

    def __init__(self, level: "IntegrityLabel.Level"):
        self.level = level

    def leq(self, other: "IntegrityLabel") -> bool:
        return self.level.value <= other.level.value

    def join(self, other: "IntegrityLabel") -> "IntegrityLabel":
        return other if self.leq(other) else self

    def meet(self, other: "IntegrityLabel") -> "IntegrityLabel":
        return self if self.leq(other) else other

    def __repr__(self):
        return f"{self.level.name}"

    @classmethod
    def trusted(cls):
        return cls(cls.Level.TRUSTED)

    @classmethod
    def untrusted(cls):
        return cls(cls.Level.UNTRUSTED)

    # --- Convenience ---
    @property
    def is_untrusted(self) -> bool:
        return self.level == IntegrityLabel.Level.UNTRUSTED


T = TypeVar("T")  # The type of elements in the base set


class PowersetLattice(Lattice, Generic[T]):
    """Powerset lattice ordered by subset inclusion."""

    def __init__(self, subset: FrozenSet[T], universe: FrozenSet[T]):
        if not subset.issubset(universe):
            raise ValueError("Subset must be within the universe.")
        self.subset = subset
        self.universe = universe

    def leq(self, other: "PowersetLattice[T]") -> bool:
        return self.subset.issubset(other.subset)

    def join(self, other: "PowersetLattice[T]") -> "PowersetLattice[T]":
        return PowersetLattice(self.subset.union(other.subset), self.universe)

    def meet(self, other: "PowersetLattice[T]") -> "PowersetLattice[T]":
        return PowersetLattice(self.subset.intersection(other.subset), self.universe)

    def __repr__(self):
        return f"Powerset({{{', '.join(map(str, self.subset))}}})"

    @classmethod
    def bottom(cls, universe: FrozenSet[T]) -> "PowersetLattice[T]":
        return cls(frozenset(), universe)

    @classmethod
    def top(cls, universe: FrozenSet[T]) -> "PowersetLattice[T]":
        return cls(universe, universe)


L1 = TypeVar("L1", bound=Lattice)
L2 = TypeVar("L2", bound=Lattice)


class ProductLabel(Lattice, Generic[L1, L2]):
    """Product of two lattices, ordered componentwise."""

    def __init__(self, left: L1, right: L2):
        self.left = left
        self.right = right

    def leq(self, other: "ProductLabel[L1, L2]") -> bool:
        return self.left <= other.left and self.right <= other.right

    def join(self, other: "ProductLabel[L1, L2]") -> "ProductLabel[L1, L2]":
        return ProductLabel(self.left.join(other.left), self.right.join(other.right))

    def meet(self, other: "ProductLabel[L1, L2]") -> "ProductLabel[L1, L2]":
        return ProductLabel(self.left.meet(other.left), self.right.meet(other.right))

    def __repr__(self):
        return f"({self.left}, {self.right})"


L = TypeVar("L", bound="Lattice")


class InverseLattice(Lattice, Generic[L]):
    """The inverse of a lattice (order reversed). Wrapping the powerset lattice gives a
    confidentiality lattice ordered by *readers*: a smaller reader set is more secret."""

    def __init__(self, inner: L):
        self.inner = inner

    def leq(self, other: "InverseLattice[L]") -> bool:
        return other.inner.leq(self.inner)  # Invert the order

    def join(self, other: "InverseLattice[L]") -> "InverseLattice[L]":
        return InverseLattice(self.inner.meet(other.inner))  # Invert operation

    def meet(self, other: "InverseLattice[L]") -> "InverseLattice[L]":
        return InverseLattice(self.inner.join(other.inner))  # Invert operation

    def __repr__(self):
        return f"Inverse({repr(self.inner)})"

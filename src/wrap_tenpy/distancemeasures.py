import numpy as np
from tenpy.linalg import np_conserved as npc


def hilbertschmidt_distance(a, b):
    """Return the squared Hilbert-Schmidt distance between TenPy operators."""
    labels = sorted(a.get_leg_labels())
    if labels != sorted(b.get_leg_labels()):
        raise ValueError("operators must have matching leg labels")

    ket_labels = sorted(label for label in labels if not label.endswith("*"))
    bra_labels = sorted(label for label in labels if label.endswith("*"))

    traced = npc.tensordot(a - b, a - b, axes=(bra_labels, ket_labels))
    for ket_label, bra_label in zip(ket_labels, bra_labels):
        traced = npc.trace(traced, leg1=ket_label, leg2=bra_label)

    return np.real(traced)

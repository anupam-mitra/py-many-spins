import qutip
from qutip_qip.operations import expand_operator


def _pauli_operator(operator):
    if operator == "x":
        return qutip.sigmax()
    if operator == "y":
        return qutip.sigmay()
    if operator == "z":
        return qutip.sigmaz()
    raise ValueError("unsupported Pauli operator %r" % (operator,))


class QutipSpinHalfPauliHamiltonian:
    """QuTiP Hamiltonian adapter for a neutral spin-half Pauli model."""

    def __init__(self, model):
        self.model = model
        self.dim_local = [2 for _ in range(model.n_sites)]

    def construct_hamiltonian_qutip(self, n_spins=None):
        if n_spins is None:
            n_spins = self.model.n_sites
        if n_spins != self.model.n_sites:
            raise ValueError(
                "n_spins=%d does not match model.n_sites=%d"
                % (n_spins, self.model.n_sites)
            )

        dims = [2] * self.model.n_sites
        h_local_terms = []
        h_interact_terms = []

        for coefficient, operator, site in self.model.expanded_local_terms():
            h_local_terms.append(
                coefficient
                * expand_operator(
                    _pauli_operator(operator),
                    targets=(site,),
                    dims=dims,
                )
            )

        for coefficient, operators, left, right in self.model.expanded_two_site_terms():
            local_operator = qutip.tensor(
                _pauli_operator(operators[0]),
                _pauli_operator(operators[1]),
            )
            h_interact_terms.append(
                coefficient
                * expand_operator(
                    local_operator,
                    targets=(left, right),
                    dims=dims,
                )
            )

        hamiltonian_terms = h_local_terms + h_interact_terms
        self.h_local_terms = h_local_terms
        self.h_interact_terms = h_interact_terms
        self.hamiltonian_terms = hamiltonian_terms
        return hamiltonian_terms


def to_qutip_hamiltonian(model):
    """Return an object compatible with the existing QuTiP evolution wrappers."""
    return QutipSpinHalfPauliHamiltonian(model)


def to_qutip_terms(model, n_spins=None):
    """Return expanded QuTiP Hamiltonian terms for ``model``."""
    return to_qutip_hamiltonian(model).construct_hamiltonian_qutip(n_spins)

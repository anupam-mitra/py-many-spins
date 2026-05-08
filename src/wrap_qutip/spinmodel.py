import qutip


def _pauli_operator(operator):
    if operator == "x":
        return qutip.sigmax()
    if operator == "y":
        return qutip.sigmay()
    if operator == "z":
        return qutip.sigmaz()
    raise ValueError("unsupported Pauli operator %r" % (operator,))


def _embed_spinhalf_operator(n_sites, site_operators):
    # This helper is intentionally narrow for spin-half Pauli chains. If future
    # models need general subsystem embedding, prefer qutip-qip expand_operator.
    operators_by_site = dict(site_operators)
    if len(operators_by_site) != len(site_operators):
        raise ValueError("multiple operators on the same site are not supported")
    if any(site < 0 or site >= n_sites for site in operators_by_site):
        raise ValueError("operator site exceeds n_sites")

    factors = []
    for site in range(n_sites):
        factors.append(operators_by_site.get(site, qutip.qeye(2)))

    return qutip.tensor(factors)


def _zero_hamiltonian(n_sites):
    return 0 * qutip.tensor([qutip.qeye(2) for _ in range(n_sites)])


def to_qutip_hamiltonian(model):
    """Convert a neutral spin-half Pauli model to a QuTiP Hamiltonian."""
    terms = []

    for coefficient, operator, site in model.expanded_local_terms():
        terms.append(
            coefficient
            * _embed_spinhalf_operator(
                model.n_sites,
                [(site, _pauli_operator(operator))],
            )
        )

    for coefficient, operators, left, right in model.expanded_two_site_terms():
        terms.append(
            coefficient
            * _embed_spinhalf_operator(
                model.n_sites,
                [
                    (left, _pauli_operator(operators[0])),
                    (right, _pauli_operator(operators[1])),
                ],
            )
        )

    if not terms:
        return _zero_hamiltonian(model.n_sites)

    return sum(terms[1:], terms[0])

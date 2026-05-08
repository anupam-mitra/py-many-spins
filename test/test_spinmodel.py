import numpy as np
import pytest

from manybody_util.spinmodel import (
    SpinHalfPauliModel,
    TwoSiteTerm,
    WeightedEdge,
    nearest_neighbor_edges_1d,
    tilted_field_ising_1d,
)


def test_nearest_neighbor_edges_1d():
    open_edges = nearest_neighbor_edges_1d(4, bc="open")
    assert [(edge.left, edge.right, edge.weight) for edge in open_edges] == [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
    ]

    periodic_edges = nearest_neighbor_edges_1d(4, bc="periodic")
    assert [(edge.left, edge.right, edge.weight) for edge in periodic_edges] == [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 0, 1.0),
    ]


def test_tilted_field_ising_1d_expands_terms():
    model = tilted_field_ising_1d(
        n_sites=4,
        j_xx=0.1,
        b_z=1.0,
        b_x=0.15,
        bc="open",
    )

    assert model.n_sites == 4
    assert model.metadata["family"] == "tilted_field_ising_1d"
    assert model.metadata["bc"] == "open"

    local_terms = list(model.expanded_local_terms())
    assert len(local_terms) == 8
    assert local_terms[:4] == [
        (1.0, "z", 0),
        (1.0, "z", 1),
        (1.0, "z", 2),
        (1.0, "z", 3),
    ]
    assert local_terms[4:] == [
        (0.15, "x", 0),
        (0.15, "x", 1),
        (0.15, "x", 2),
        (0.15, "x", 3),
    ]

    two_site_terms = list(model.expanded_two_site_terms())
    assert two_site_terms == [
        (0.1, ("x", "x"), 0, 1),
        (0.1, ("x", "x"), 1, 2),
        (0.1, ("x", "x"), 2, 3),
    ]


def test_qutip_adapter_constructs_hamiltonian_terms():
    qutip = pytest.importorskip("qutip")

    from wrap_qutip.spinmodel import to_qutip_hamiltonian

    model = tilted_field_ising_1d(3, j_xx=0.1, b_z=1.0, b_x=0.15)
    hamiltonian = to_qutip_hamiltonian(model)

    identity = qutip.qeye(2)
    sigmax = qutip.sigmax()
    sigmaz = qutip.sigmaz()
    expected = (
        qutip.tensor(sigmaz, identity, identity)
        + qutip.tensor(identity, sigmaz, identity)
        + qutip.tensor(identity, identity, sigmaz)
        + 0.15 * qutip.tensor(sigmax, identity, identity)
        + 0.15 * qutip.tensor(identity, sigmax, identity)
        + 0.15 * qutip.tensor(identity, identity, sigmax)
        + 0.1 * qutip.tensor(sigmax, sigmax, identity)
        + 0.1 * qutip.tensor(identity, sigmax, sigmax)
    )

    assert hamiltonian.dims == [[2, 2, 2], [2, 2, 2]]
    assert np.allclose((hamiltonian - expected).full(), 0.0)


def test_qutip_adapter_embeds_noncontiguous_two_site_terms():
    qutip = pytest.importorskip("qutip")

    from wrap_qutip.spinmodel import to_qutip_hamiltonian

    model = SpinHalfPauliModel(
        n_sites=3,
        two_site_terms=(
            TwoSiteTerm(0.7, ("x", "z"), (WeightedEdge(2, 0),)),
        ),
    )
    hamiltonian = to_qutip_hamiltonian(model)
    expected = 0.7 * qutip.tensor(qutip.sigmaz(), qutip.qeye(2), qutip.sigmax())

    assert hamiltonian.dims == [[2, 2, 2], [2, 2, 2]]
    assert np.allclose((hamiltonian - expected).full(), 0.0)


def test_qutip_sesolve_state_history_and_marginal_helpers():
    pytest.importorskip("qutip")

    from wrap_qutip.spinmodel import to_qutip_hamiltonian
    from wrap_qutip.timeevolution import (
        local_marginal_density_matrix,
        manyspin_product_state,
        solve_state_history,
    )

    model = tilted_field_ising_1d(2, j_xx=0.0, b_z=0.2, b_x=0.0)
    initial_state = manyspin_product_state(2, np.pi / 2, 0.0)
    tlist = np.array([0.0, 0.1, 0.3])

    df_states, states = solve_state_history(
        to_qutip_hamiltonian(model),
        initial_state,
        tlist,
        metadata={"bonddim": None},
    )

    assert len(states) == len(tlist)
    assert np.allclose(df_states["time"], tlist)
    assert states[0].dims == initial_state.dims
    assert np.isclose(states[-1].norm(), 1.0)

    rho = local_marginal_density_matrix(states[-1], (0,))
    assert rho.dims == [[2], [2]]
    assert np.isclose(rho.tr(), 1.0)


def test_quspin_adapter_constructs_static_terms():
    pytest.importorskip("quspin")

    from wrap_quspin.spinmodel import (
        to_quspin_basis,
        to_quspin_hamiltonian,
        to_quspin_static_terms,
    )

    model = tilted_field_ising_1d(3, j_xx=0.1, b_z=1.0, b_x=0.15)
    static_terms = dict(to_quspin_static_terms(model))

    assert static_terms["z"] == [[1.0, 0], [1.0, 1], [1.0, 2]]
    assert static_terms["x"] == [[0.15, 0], [0.15, 1], [0.15, 2]]
    assert static_terms["xx"] == [[0.1, 0, 1], [0.1, 1, 2]]

    basis = to_quspin_basis(model)
    hamiltonian = to_quspin_hamiltonian(model, basis=basis)
    assert hamiltonian.Ns == 2**model.n_sites
    assert hamiltonian.toarray().shape == (2**model.n_sites, 2**model.n_sites)


def test_quspin_adapter_embeds_noncontiguous_two_site_terms():
    pytest.importorskip("quspin")

    from wrap_quspin.spinmodel import to_quspin_static_terms

    model = SpinHalfPauliModel(
        n_sites=3,
        two_site_terms=(
            TwoSiteTerm(0.7, ("x", "z"), (WeightedEdge(2, 0),)),
        ),
    )

    assert dict(to_quspin_static_terms(model))["xz"] == [[0.7, 2, 0]]


def test_quspin_exact_evolution_and_marginal_helpers():
    pytest.importorskip("quspin")

    from wrap_quspin.spinmodel import to_quspin_basis, to_quspin_hamiltonian
    from wrap_quspin.timeevolution import (
        local_marginal_density_matrix,
        manyspin_product_state,
        solve_state_history,
    )

    model = tilted_field_ising_1d(2, j_xx=0.0, b_z=0.2, b_x=0.0)
    basis = to_quspin_basis(model)
    initial_state = manyspin_product_state(2, np.pi / 2, 0.0, basis=basis)
    tlist = np.array([0.0, 0.1, 0.3])

    df_states, states = solve_state_history(
        to_quspin_hamiltonian(model, basis=basis),
        initial_state,
        tlist,
        metadata={"bonddim": None},
    )

    assert len(states) == len(tlist)
    assert np.allclose(df_states["time"], tlist)
    assert states[0].shape == initial_state.shape
    assert np.isclose(np.linalg.norm(states[-1]), 1.0)

    rho = local_marginal_density_matrix(states[-1], (0,), basis)
    assert rho.shape == (2, 2)
    assert np.isclose(np.trace(rho), 1.0)


def test_qutip_quimb_conversion_uses_modern_qobj_api():
    qutip = pytest.importorskip("qutip")
    pytest.importorskip("quimb.tensor")

    from conversions.convquimbqutip import (
        convert_qutip_ket_to_quimb_mps,
        convert_quimb_mp_to_qutip_qobj,
    )

    ket = qutip.tensor([qutip.basis(2, 0), qutip.basis(2, 1)])
    mps = convert_qutip_ket_to_quimb_mps(ket)
    reconstructed = convert_quimb_mp_to_qutip_qobj(mps)

    assert reconstructed.dims == ket.dims
    assert np.isclose(abs(ket.overlap(reconstructed)), 1.0)


def test_quimb_adapter_builds_spinham1d():
    pytest.importorskip("quimb")
    pytest.importorskip("quimb.tensor")

    from wrap_quimb.spinmodel import to_quimb_spinham1d

    model = tilted_field_ising_1d(3, j_xx=0.1, b_z=1.0, b_x=0.15)
    builder = to_quimb_spinham1d(model)

    assert builder.build_local_ham(model.n_sites) is not None


def test_quimb_tebd_wrapper_uses_modern_api():
    pytest.importorskip("quimb")
    qtn = pytest.importorskip("quimb.tensor")

    from wrap_quimb.quimbtebd import TEBDWrapper, spinhalf_state
    from wrap_quimb.spinmodel import to_quimb_spinham1d

    model = tilted_field_ising_1d(3, j_xx=0.1, b_z=1.0, b_x=0.15)
    initial_mps = qtn.MPS_product_state(
        [spinhalf_state(np.pi / 2, np.pi / 2)] * model.n_sites,
        cyclic=False,
    )
    tlist = np.array([0.0, 0.05])

    wrapper = TEBDWrapper(
        to_quimb_spinham1d(model),
        initial_mps,
        tlist,
        trotter_params={"dt": 0.05, "order": 2, "progbar": False},
        trunc_params={"chi_max": 4, "cutoff": 1e-12},
    )
    wrapper.evolve()

    df_mps, mps_list = wrapper.get_mps_history_df()
    assert len(mps_list) == len(tlist)
    assert list(df_mps["ix_time"]) == [0, 1]
    assert wrapper.tebd.split_opts["max_bond"] == 4


def test_quimb_mps_helpers():
    qtn = pytest.importorskip("quimb.tensor")

    from wrap_quimb.quimbtebd import (
        local_marginal_density_matrix,
        max_bond_dimension,
        spinhalf_state,
    )

    mps = qtn.MPS_product_state(
        [spinhalf_state(np.pi / 2, np.pi / 2)] * 3,
        cyclic=False,
    )

    rho = local_marginal_density_matrix(mps, (1,))
    assert rho.shape == (2, 2)
    assert np.allclose(np.trace(rho), 1.0)
    assert max_bond_dimension(mps) == 1


def test_tenpy_adapter_builds_direct_pauli_chain():
    pytest.importorskip("tenpy")
    from tenpy.networks.mps import MPS

    from wrap_tenpy.spinmodel import to_tenpy_model

    model = tilted_field_ising_1d(2, j_xx=0.0, b_z=1.0, b_x=0.0)
    tenpy_model = to_tenpy_model(model, conserve=None)
    psi = MPS.from_product_state(
        tenpy_model.lat.mps_sites(),
        ['up'] * model.n_sites,
        bc=tenpy_model.lat.bc_MPS,
        dtype=complex,
        unit_cell_width=tenpy_model.lat.mps_unit_cell_width,
    )

    assert len(tenpy_model.H_bond) == model.n_sites
    assert np.isclose(tenpy_model.H_MPO.expectation_value(psi), 2.0)


def test_tenpy_tebd_evolution_and_marginal_helpers():
    pytest.importorskip("tenpy")

    from wrap_tenpy.spinmodel import to_tenpy_model
    from wrap_tenpy.timeevolution import (
        local_marginal_density_matrix,
        manyspin_product_mps,
        max_bond_dimension,
        solve_mps_history,
    )

    model = tilted_field_ising_1d(2, j_xx=0.0, b_z=0.2, b_x=0.0)
    tenpy_model = to_tenpy_model(model, conserve=None)
    initial_mps = manyspin_product_mps(
        tenpy_model.lat.mps_sites(),
        np.pi / 2,
        0.0,
        bc=tenpy_model.lat.bc_MPS,
        unit_cell_width=tenpy_model.lat.mps_unit_cell_width,
    )
    tlist = np.array([0.0, 0.05])

    df_mps, mps_list = solve_mps_history(
        tenpy_model,
        initial_mps,
        tlist,
        algorithm="TEBD",
        evolution_params={"order": 2, "N_steps": 1},
        trunc_params={"chi_max": 4, "svd_min": 1e-12},
        metadata={"bonddim": 4},
    )

    assert len(mps_list) == len(tlist)
    assert np.allclose(df_mps["time"], tlist)
    assert list(df_mps["ix_time"]) == [0, 1]
    assert max_bond_dimension(mps_list[-1]) <= 4

    rho = local_marginal_density_matrix(mps_list[-1], (0,))
    assert rho.shape == (2, 2)
    assert np.isclose(np.trace(rho.to_ndarray()), 1.0)

    df_tdvp, tdvp_mps_list = solve_mps_history(
        tenpy_model,
        initial_mps,
        np.array([0.0, 0.01]),
        algorithm="TDVP",
        trunc_params={"chi_max": 4},
        metadata={"bonddim": 4},
    )
    assert len(tdvp_mps_list) == 2
    assert list(df_tdvp["ix_time"]) == [0, 1]

    df_expmpo, expmpo_mps_list = solve_mps_history(
        tenpy_model,
        initial_mps,
        np.array([0.0, 0.01]),
        algorithm="ExpMPO",
        trunc_params={"chi_max": 4, "svd_min": 1e-12},
        metadata={"bonddim": 4},
    )
    assert len(expmpo_mps_list) == 2
    assert list(df_expmpo["ix_time"]) == [0, 1]
    assert max_bond_dimension(expmpo_mps_list[-1]) <= 4

    rho_expmpo = local_marginal_density_matrix(expmpo_mps_list[-1], (0,))
    assert rho_expmpo.shape == (2, 2)
    assert np.isclose(np.trace(rho_expmpo.to_ndarray()), 1.0)

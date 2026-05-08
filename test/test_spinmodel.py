import numpy as np
import pytest

from manybody_util.spinmodel import (
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
    pytest.importorskip("qutip")
    pytest.importorskip("qutip.qip.operations")

    from wrap_qutip.spinmodel import to_qutip_hamiltonian

    model = tilted_field_ising_1d(3, j_xx=0.1, b_z=1.0, b_x=0.15)
    hamiltonian = to_qutip_hamiltonian(model).construct_hamiltonian_qutip()

    assert len(hamiltonian) == 8
    assert hamiltonian[0].dims == [[2, 2, 2], [2, 2, 2]]


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


def test_tenpy_adapter_builds_model():
    pytest.importorskip("tenpy")

    from wrap_tenpy.spinmodel import to_tenpy_model

    model = tilted_field_ising_1d(3, j_xx=0.1, b_z=1.0, b_x=0.15)

    assert to_tenpy_model(model) is not None

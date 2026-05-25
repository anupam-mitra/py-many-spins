import numpy as np
import itertools
import pytest

tenpy = pytest.importorskip("tenpy")
import tenpy.networks.site


from manybody_backends.tenpy.distancemeasures import hilbertschmidt_distance

################################################################################
def test_squared_hilbertschmidt_distance_onespin_rho() -> None:
    """
    Test the squared Hilbert-Schmidt distance for single-spin density matrices.

    Verifies the distance between various pure and mixed states of a
    single spin-1/2 system.
    """
    site:tenpy.networks.site.SpinHalfSite \
        = tenpy.networks.site.SpinHalfSite(conserve=None)

    op_id = site.get_op("Id")
    op_x = site.get_op("Sigmax")
    op_y = site.get_op("Sigmay")
    op_z = site.get_op("Sigmaz")

    rho_maxmix = op_id * 0.5
    rho_upz = (op_id + op_x) * 0.5
    rho_dnz = (op_id - op_x) * 0.5
    rho_upy = (op_id + op_y) * 0.5
    rho_dny = (op_id - op_y) * 0.5
    rho_upx = (op_id + op_z) * 0.5
    rho_dnx = (op_id - op_z) * 0.5

    assert 0.0 == hilbertschmidt_distance(rho_upz, rho_upz)
    assert 2.0 == hilbertschmidt_distance(rho_upz, rho_dnz)
    assert 0.0 == hilbertschmidt_distance(rho_dnz, rho_dnz)

    assert 0.0 == hilbertschmidt_distance(rho_upy, rho_upy)
    assert 2.0 == hilbertschmidt_distance(rho_upy, rho_dny)
    assert 0.0 == hilbertschmidt_distance(rho_dny, rho_dny)

    assert 0.0 == hilbertschmidt_distance(rho_dnx, rho_dnx)
    assert 2.0 == hilbertschmidt_distance(rho_upx, rho_dnx)
    assert 0.0 == hilbertschmidt_distance(rho_upx, rho_upx)

    assert 1.0 == hilbertschmidt_distance(rho_upz, rho_upx)
    assert 1.0 == hilbertschmidt_distance(rho_dnz, rho_dnx)
    assert 1.0 == hilbertschmidt_distance(rho_upz, rho_dnx)
    assert 1.0 == hilbertschmidt_distance(rho_upx, rho_dnz)

    assert 1.0 == hilbertschmidt_distance(rho_upz, rho_upy)
    assert 1.0 == hilbertschmidt_distance(rho_dnz, rho_dny)
    assert 1.0 == hilbertschmidt_distance(rho_upz, rho_dny)
    assert 1.0 == hilbertschmidt_distance(rho_upy, rho_dnz)

    assert 1.0 == hilbertschmidt_distance(rho_upx, rho_upy)
    assert 1.0 == hilbertschmidt_distance(rho_dnx, rho_dny)
    assert 1.0 == hilbertschmidt_distance(rho_upx, rho_dny)
    assert 1.0 == hilbertschmidt_distance(rho_upy, rho_dnx)

    assert 0.5 == hilbertschmidt_distance(rho_maxmix, rho_upx)
    assert 0.5 == hilbertschmidt_distance(rho_maxmix, rho_dnx)
    assert 0.5 == hilbertschmidt_distance(rho_maxmix, rho_upy)
    assert 0.5 == hilbertschmidt_distance(rho_maxmix, rho_dny)
    assert 0.5 == hilbertschmidt_distance(rho_maxmix, rho_upz)
    assert 0.5 == hilbertschmidt_distance(rho_maxmix, rho_dnz)
################################################################################ 

################################################################################
def test_squared_hilbertschmidt_distance_onespin_operator_basis() -> None:
    """
    Test the squared Hilbert-Schmidt distance for single-spin operators in the Pauli basis.

    Verifies that the distance between different Pauli operators matches
    the expected theoretical values.
    """
    site:tenpy.networks.site.SpinHalfSite \
        = tenpy.networks.site.SpinHalfSite(conserve=None)

    op_id = site.get_op("Id")
    op_x = site.get_op("Sigmax")
    op_y = site.get_op("Sigmay")
    op_z = site.get_op("Sigmaz")

    op_list = [op_id, op_x, op_y, op_z]

    for op_a, op_b in itertools.combinations_with_replacement(
            op_list, 2):
        if op_a == op_b:
            assert 0.0 == hilbertschmidt_distance(op_a, op_b)
        else:
            assert 4.0 == hilbertschmidt_distance(op_a, op_b)
################################################################################

################################################################################
def test_squared_hilbertschmidt_distance_twospin_operator_basis() -> None:
    """
    Test the squared Hilbert-Schmidt distance for two-spin operators.

    Verifies the distance between various tensor product Pauli operators
    for a two-site system.
    """
    site:tenpy.networks.site.SpinHalfSite \
        = tenpy.networks.site.SpinHalfSite(conserve=None)

    op_id = site.get_op("Id")
    op_x = site.get_op("Sigmax")
    op_y = site.get_op("Sigmay")
    op_z = site.get_op("Sigmaz")

    op_id0 = op_id.replace_labels(["p", "p*"], ["p0", "p0*"])
    op_id1 = op_id.replace_labels(["p", "p*"], ["p1", "p1*"])

    op_x0 = op_x.replace_labels(["p", "p*"], ["p0", "p0*"])
    op_x1 = op_x.replace_labels(["p", "p*"], ["p1", "p1*"])
    
    op_y0 = op_y.replace_labels(["p", "p*"], ["p0", "p0*"])
    op_y1 = op_y.replace_labels(["p", "p*"], ["p1", "p1*"])
    
    op_z0 = op_z.replace_labels(["p", "p*"], ["p0", "p0*"])
    op_z1 = op_z.replace_labels(["p", "p*"], ["p1", "p1*"])

    op_id0_id1 = tenpy.linalg.np_conserved.outer(op_id0, op_id1)

    op_id0_x1 = tenpy.linalg.np_conserved.outer(op_id0, op_x1)
    op_id0_y1 = tenpy.linalg.np_conserved.outer(op_id0, op_y1)
    op_id0_z1 = tenpy.linalg.np_conserved.outer(op_id0, op_z1)

    op_x0_id1 = tenpy.linalg.np_conserved.outer(op_x0, op_id1)
    op_y0_id1 = tenpy.linalg.np_conserved.outer(op_y0, op_id1)
    op_z0_id1 = tenpy.linalg.np_conserved.outer(op_z0, op_id1)

    op_x0_x1 = tenpy.linalg.np_conserved.outer(op_x0, op_x1)
    op_x0_y1 = tenpy.linalg.np_conserved.outer(op_x0, op_y1)
    op_x0_z1 = tenpy.linalg.np_conserved.outer(op_x0, op_z1)

    op_y0_x1 = tenpy.linalg.np_conserved.outer(op_y0, op_x1)
    op_y0_y1 = tenpy.linalg.np_conserved.outer(op_y0, op_y1)
    op_y0_z1 = tenpy.linalg.np_conserved.outer(op_y0, op_z1)

    op_z0_x1 = tenpy.linalg.np_conserved.outer(op_z0, op_x1)
    op_z0_y1 = tenpy.linalg.np_conserved.outer(op_z0, op_y1)
    op_z0_z1 = tenpy.linalg.np_conserved.outer(op_z0, op_z1)

    op_list = [\
     op_id0_id1, \
     op_id0_x1, op_id0_y1, op_id0_z1, \
     op_x0_id1, op_y0_id1, op_z0_id1, \
     op_x0_x1, op_x0_y1, op_x0_z1, \
     op_y0_x1, op_y0_y1, op_y0_z1, \
     op_z0_x1, op_z0_y1, op_z0_z1,]

    for op_a, op_b in itertools.combinations_with_replacement(
            op_list, 2):
        if op_a == op_b:
            assert 0.0 == hilbertschmidt_distance(op_a, op_b)
        else:
            assert 8.0 == hilbertschmidt_distance(op_a, op_b)
################################################################################

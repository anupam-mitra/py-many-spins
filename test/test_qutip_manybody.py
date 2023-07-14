import sys
import qutip

sys.path.append("../src")

import wrap_qutip

from wrap_qutip import manybodystateevolve

from manybody_util import interactgraphs

def test_twolocal_hamiltonian():

    nSpins = 8
    jzz = 1.0
    bz = 1.0
    bx = 1.0

    interact_graph = interactgraphs.Interaction1DNearest(nSpins, bc='open')

    h = manybodystateevolve.UniformTwoBodyInteraction(
        [(qutip.sigmaz(), qutip.sigmaz())],
        [qutip.sigmax(), qutip.sigmaz()],
        [jzz], [bx, bz],
        interact_graph)

    h.construct_hamiltonian_qutip(nSpins)

    h = manybodystateevolve.UniformTwoBodyInteraction(
        [(qutip.spin_Jz(1), qutip.spin_Jz(1))],
        [qutip.spin_Jx(1), qutip.spin_Jz(1)],
        [jzz], [bx, bz],
        interact_graph)

    h.construct_hamiltonian_qutip(nSpins)
    


if __name__ == '__main__':
    test_twolocal_hamiltonian()
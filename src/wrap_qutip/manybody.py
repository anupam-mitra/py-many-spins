
class ManyBodySystem:
    """
    Class representing a many body system

    Parameters
    ----------
    num_dof: int
        Number of degrees of freedom

    dim_local: iterable
        Local dimension of each degree of freedom
    """

    def __init__(self, num_dof, dim_local):
        self.num_dof = num_dof
        self.dim_local = dim_local


class ManyBodyHamiltonian:
    """
    Represents a many body Hamiltonian
    """

    def __init__(self,):
        pass
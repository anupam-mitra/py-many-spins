import pickle

################################################################################
class StateRecord:
    """Represents a time evolution record of states, providing functionality for
    saving and loading records.
    """

    def __init__(self, timelist, states, params):
        self.timelist = timelist
        self.states = states

        self.params = params
        self.store_params = store_params

    def add_state(self, t, state):
        """Add time `t` and state `state` to record"""
        self.timelist.append(t)
        self.states.append(state)

    def add_states(self, times, states):
        """Add times `times` and states `states` to record"""
        self.timelist += times
        self.states += states

    def save_pkl(self, filepath):
        """Save timelist and state to pickled objects in file `filepath`"""

        with open(filepath, "wb") as pklout:
            pickle.dump(self.timelist, filepath)

            pickle.dump(self.states, filepath)

    def load_pkl(self, filepath):
        """Load timelist and state from pickled obijects in in file `filepath`
        """

        with open(filepath, "wb") as pklout:
            pickle.load(pklout)

            pickle.dump(self.states, filepath)


    def save_df(self, filepath):
        """Save timelist and state to pickle pandas.`DataFrame`"""
        pass

################################################################################
class StateRecordCalculation:
    """Represents a calculation on a record of states
    """

    def __init__(self, record, function, arguments):
        self.record = record
        self.function = function
        self.arguments = arguments
################################################################################
################################################################################
class StateRecordComparison:
    """Represents a comparison calculation on a record of states
    """

    def __init__(self, record_a, record_b, function, arguments):
        self.record_a = record_a
        self.record_b = record_b
        self.function = function
        self.arguments = arguments
################################################################################


################################################################################

import h5py
import tenpy.tools.hdf5_io

############################################################################################
def model_save (filename, model, tebd_params, psi_in_ref):
    '''
    Save a model used for tebd simulations to disk

    Parameters
    ----------
    filename: File name

    model: Model used for tebd

    tebd_params: Parameters used for TEBD

    psi_in_ref: Reference to the initial state

    Returns
    -------
    '''
    save_dict = {\
        'model': model, \
        'tebd_parms': tebd_params, \
        'psi_in_ref': psi_in_ref, \
    }

    with h5py.File(filename, 'w') as f:
        tenpy.tools.hdf5_io.save_to_hdf5(f, save_dict)
############################################################################################

############################################################################################
def tmps_save (filename, psi, t, model_ref, prev_ref, next_ref):
    '''
    Save a matrix product state to disk, creating and index entry.

    Parameters
    ----------

    filename: File name

    psi: State

    model_ref: Reference to the model

    prev_ref: Reference to previous state

    next_ref: Referene to next state

    t: Time

    Returns
    -------

    '''
    data = {\
        'psi': psi, \
        't': t, \
        'model_ref': model_ref, \
        'prev_ref': prev_ref, \
        'next_ref': next_ref,
    }

    with h5py.File(filename, 'w') as f:
        tenpy.tools.hdf5_io.save_to_hdf5(f, data)

############################################################################################

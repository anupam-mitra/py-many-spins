import uuid

def uuid_gen (mode='hex'):
    '''
    Generates a uuid to uniquely identify an object like
    a simulation, a hamiltonian, etc.
    '''

    uuid_current = uuid.uuid4()
    
    if mode == 'str':
        uuid_str = str(uuid_current)
    elif mode == 'hex':
        uuid_str = uuid_current.hex
    else:
        uuid_str = uuid_current.hex

    return uuid_str
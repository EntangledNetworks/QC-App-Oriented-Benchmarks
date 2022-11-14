import numpy as np

import qiskit.providers.aer.noise as noise

def c_noise_2qb(fid):
    '''
    Currently implements dephasing.
    '''
    p = np.sqrt((5*fid-1)/4)
    pZ = np.array([
        [1,0],
        [0,-1]
    ])
    id2 = np.eye(2)
    op1 = (np.kron(id2, id2), p**2)
    op2 = (np.kron(id2, pZ ), p*(1-p))
    op3 = (np.kron(pZ,  id2), p*(1-p))
    op4 = (np.kron(pZ,  pZ ), (1-p)**2)
    
    return noise.mixed_unitary_error([op1,op2,op3,op4])

def c_noise_1qb(fid):
    '''
    Currently implements dephasing.
    '''
    # First convert from fid to prob.
    p = (3*fid-1)/2
    
    pZ = np.array([
        [1,0],
        [0,-1]
    ])
    id2 = np.eye(2)
    op1 = (id2, p)
    op2 = (pZ , (1-p))
    
    return noise.mixed_unitary_error([op1,op2])
    
def get_noise(dqpu, p1qb=0.9995, p2qb=0.996, p_epr=0.97, p_spam=0.0039):
    '''
    Generate QISkit noise objects for a generic monolithic as well as D-QPU backends.
    '''
    qb_int = dqpu.interface_qubits()
    qb_comp = dqpu.comp_qubits()
    qb_all = dqpu.qubits()
    
    # Define errors
    gamma_spam = p_spam # SPAM error rate

    # Param set for "NORMAL" runs.
    gamma_native_1qb = 2*(1-p1qb) # 1-qb gate fidelity
    gamma_native_cx = 4*(1-p2qb)/3 # CX gate fidelity
    gamma_native_swap = 4*(1-p2qb**3)/3 # Swap gate fidelity

    gamma_epr = 4*(1-p_epr)/3 # EPR state fidelity

    # Define noise model. Note that thermal noise is being added in rebuild_with_delays().
    noise_model_dqpu = noise.NoiseModel()
    noise_model_mono = noise.NoiseModel()

    noise_model_dqpu.add_all_qubit_quantum_error(noise.depolarizing_error(gamma_native_1qb,1), ['rx', 'ry', 'rz', 'h', 'x', 'z'])
    for i in range(len(qb_all)):
        for j in range(i+1,len(qb_all)):
            qargs = [qb_all[i], qb_all[j]]
            qb1_isIntercon = qargs[0] in qb_comp
            qb2_isIntercon = qargs[1] in qb_comp
            if qb1_isIntercon or qb2_isIntercon:
                noise_model_dqpu.add_quantum_error(noise.depolarizing_error(gamma_native_cx,2), ['cx'], qargs)
                noise_model_dqpu.add_quantum_error(noise.depolarizing_error(gamma_native_swap,2), ['swap'], qargs)
                qargs.reverse()
                noise_model_dqpu.add_quantum_error(noise.depolarizing_error(gamma_native_cx,2), ['cx'], qargs)
                noise_model_dqpu.add_quantum_error(noise.depolarizing_error(gamma_native_swap,2), ['swap'], qargs)
            else:
                noise_model_dqpu.add_quantum_error(noise.depolarizing_error(gamma_epr, 2), ['cx'], qargs)
                qargs.reverse()
                noise_model_dqpu.add_quantum_error(noise.depolarizing_error(gamma_epr, 2), ['cx'], qargs)

    noise_model_mono.add_all_qubit_quantum_error(noise.depolarizing_error(gamma_native_1qb,1), ['rx', 'ry', 'rz', 'h', 'x', 'z'])
    noise_model_mono.add_all_qubit_quantum_error(noise.depolarizing_error(gamma_native_cx,2), ['cx'])
    noise_model_mono.add_all_qubit_quantum_error(noise.depolarizing_error(gamma_native_swap,2), ['swap'])

    return noise_model_dqpu, noise_model_mono

def get_noise_custom(dqpu, get_customnoise_1qb, get_customnoise_2qb, p1qb=0.9995, p2qb=0.996, p_epr=0.97, p_spam=0.0039):
    '''
    Like get_noise() above, but using custom noise channels instead of a built-in QISkit channel.

    Args get_customnoise_[1/2]qb are functions that return a mixed_unitary_error object.
    '''
    qb_int = dqpu.interface_qubits()
    qb_comp = dqpu.comp_qubits()
    qb_all = dqpu.qubits()

    gamma_epr = 4*(1-p_epr)/3 # EPR state fidelity

    # Define noise model. Note that thermal noise is being added in rebuild_with_delays().
    noise_model_dqpu = noise.NoiseModel()
    noise_model_mono = noise.NoiseModel()

    noise_model_dqpu.add_all_qubit_quantum_error(get_customnoise_1qb(p1qb), ['rx', 'ry', 'rz', 'h', 'x', 'z'])
    for i in range(len(qb_all)):
        for j in range(i+1,len(qb_all)):
            qargs = [qb_all[i], qb_all[j]]
            qb1_isIntercon = qargs[0] in qb_comp
            qb2_isIntercon = qargs[1] in qb_comp
            if qb1_isIntercon or qb2_isIntercon:
                noise_model_dqpu.add_quantum_error(get_customnoise_2qb(p2qb), ['cx'], qargs)
                noise_model_dqpu.add_quantum_error(get_customnoise_2qb(p2qb**3), ['swap'], qargs)
                qargs.reverse()
                noise_model_dqpu.add_quantum_error(get_customnoise_2qb(p2qb), ['cx'], qargs)
                noise_model_dqpu.add_quantum_error(get_customnoise_2qb(p2qb**3), ['swap'], qargs)
            else:
                noise_model_dqpu.add_quantum_error(noise.depolarizing_error(gamma_epr, 2), ['cx'], qargs)
                qargs.reverse()
                noise_model_dqpu.add_quantum_error(noise.depolarizing_error(gamma_epr, 2), ['cx'], qargs)

    noise_model_mono.add_all_qubit_quantum_error(get_customnoise_1qb(p1qb), ['rx', 'ry', 'rz', 'h', 'x', 'z'])
    noise_model_mono.add_all_qubit_quantum_error(get_customnoise_2qb(p2qb), ['cx'])
    noise_model_mono.add_all_qubit_quantum_error(get_customnoise_2qb(p2qb**3), ['swap'])

    return noise_model_dqpu, noise_model_mono
import time

import numpy as np

import qiskit

def get_bounding_aq(width,depth):
    '''
    Returns bounding AQ.
    '''
    w_depth = np.floor(np.sqrt(depth))
    return max(w_depth, width)

def check_aq(metrics):
    '''
    Check whether runs represented in 'metrics' all
    pass the AQ threshold.

    Returns a 2-tuple, (passfail, aq). If passfail=False,
    then 'aq' is the bounding value corresponding to dimensions
    of the first failed circuit. If passfail=True, then all
    runs passed the AQ threshold and 'aq' is set to -1.
    '''
    aq_min = -1
    passes = True

    for gr in metrics.group_metrics['groups']:
        passes = True # Track if all runs pass threshold.
        c_width = int(gr) # Circuit width

        for run in metrics.circuit_metrics[gr]:
            # Data for a given run.
            dat = metrics.circuit_metrics[gr][run]
            
            # 2-qubit gate count
            c_depth = dat['tr_n2q']

            # Calculate score with spread
            fid = dat['fidelity']
            spread = np.sqrt( fid*(1-fid)/num_shots )
            score = fid-spread
            if score < metrics.aq_cutoff:
                passes = False
                aq_min = get_bounding_aq(c_width, c_depth)
                break
        
        if not passes:
            break
    
    return passes, aq_min

def rebuild_with_delays(qc, dqpu, delays):
    '''
    Add delay gates into circuit to reflect finite interconnect times.
    Then, insert noise ops targeting those delay gates.

    Note:
        Under stall (waiting for interconnect), delays are added to all rails,
        simulating idle-time on all qubits. However, this delay is calculated
        as the difference between time-to-EPR vs the LONGEST active rail op.
        Ergo, this *could* underestimate wait times on other less-active rails.

        For future UPGRADE, we should add more granular delays calculated on
        a per-rail basis.
    '''
    gate_times = delays['gtime']
    ictime = delays['ictime']
    t1 = delays['t1']
    t2 = delays['t2']
    
    # Process circuit to insert interconnect stalls
    dag = qiskit.converters.circuit_to_dag(qc)
    dag_out = dag.copy_empty_like()
    time_since_interconnect_use = np.inf # Assume first interconnect is free
    total_stall = 0.0 # Total stall time for interconnect uses in seconds

    # Loop through circuit layers, find interconnect uses, count time in between them, and inject idle gates as appropriate
    for layer in dag.layers():
        layer_duration = 0.0
        for node in layer['graph'].op_nodes():
            gate = node.op
            qargs = node.qargs
            cargs = node.cargs

            isTwoQubit = (len(qargs) == 2)
            isInterface0 = (dag.qubits[dqpu.interface_qubits()[0]] in qargs)
            isInterface1 = (dag.qubits[dqpu.interface_qubits()[1]] in qargs)
                
            # Test if gate operation is on interconnect
            if isTwoQubit and isInterface0 and isInterface1:
                stall = max(0.0, ictime - time_since_interconnect_use)
                if stall > 0.0:
                    total_stall += stall
                    # Add stall operation to every qubit
                    dag_out.apply_operation_back(qiskit.circuit.Barrier(dag_out.num_qubits()), dag_out.qubits, [])
                    for qb in dag_out.qubits:
                        dag_out.apply_operation_back(qiskit.circuit.Delay(stall, 's'), [qb], [])
                    dag_out.apply_operation_back(qiskit.circuit.Barrier(dag_out.num_qubits()), dag_out.qubits, [])
                # Reset timer
                time_since_interconnect_use = 0.0
            elif gate.name in gate_times:
                layer_duration = max(layer_duration, gate_times[gate.name])
            
            # Add gate to output circuit
            dag_out.apply_operation_back(gate, qargs, cargs)

        time_since_interconnect_use += layer_duration

    # Apply thermal noise pass
    # Internally, qiskit uses the compiler pass to add thermal noise. The compiler pass inserts simulated noise after each gate.
    noise_pass = qiskit.providers.aer.noise.RelaxationNoisePass([t1]*len(dag_out.qubits), [t2]*len(dag_out.qubits), op_types=qiskit.circuit.Delay)
    dag_out = noise_pass.run(dag_out)
    
    qc = qiskit.converters.dag_to_circuit(dag_out)
    qc.total_stall = total_stall

    return qc

# >>>>> MCMR / SPAM error support functions >>>>>
def get_kraus_bitflip(prob, mode='x'):
    '''
    Returns a Kraus channel object, implementing
    incoherent bit-flip with probability 'prob'.
    '''
    if mode=='x':
        k1 = np.array([
            [1,0],
            [0,1]
        ])*np.sqrt(1-prob)
        k2 = np.array([
            [0,1],
            [1,0]
        ])*np.sqrt(prob)
    
    if mode=='z':
        k1 = np.array([
            [1,0],
            [0,1]
        ])*np.sqrt(1-prob)
        k2 = np.array([
            [1,0],
            [0,-1]
        ])*np.sqrt(prob)
    
    op = Kraus([k1, k2]).to_instruction()

    return op

def rebuild_mcmr_spam(qc, prob):
    '''
    Rebuilds circuit 'qc', inserting ancillary registers
    to implement bit-flip errors on a classical bit, in
    order to simulate SPAM errors during MCMR.
    '''
    creg_main = get_main_creg(qc)

    qc_out = qc.copy()
    qc_out.data = []
    for dat in qc.data:
        qc_out.data.append(dat)
        
        op = dat[0]
        qarg = dat[1][0]
        
        isOp = (op.name=='x') or (op.name=='z')
        isConditioned = not (op.condition==None)
        if isOp and isConditioned:
            carg = op.condition_bits[0]
            krausop = get_kraus_bitflip(prob, mode=op.name)

            # Check that this is NOT conditioned on main register.
            if not carg in creg_main:
                qc_out.append(krausop, [qarg])
    
    return qc_out
# <<<<< MCMR / SPAM error support functions <<<<<
class ResultMock:
    def __init__(self, result, hasancillas=False):
        """
        Create a result object containing only output data counts;
        classical interconnect cbits values are stripped.
        
        If hasancillas==False, circuit run is a "bare" circuit with no ancillas;
        and the stripping process is skipped. Otherwise 
        """
        self.res = result
        self._counts = self._strip_ic_from_counts(result.get_counts(), hasancillas=hasancillas)
        self._dict = (result.results)[0].to_dict()

    def _strip_ic_from_counts(self, counts, hasancillas=False):
        '''
        Utility to strip ancilla / interconnect cregs from Aer outputs.

        We ASSUME any ancilla cregs WON'T show up past the 2nd character
        in bit-strings that key any dict. of output counts. In other words
        we ASSUME the ancilla that we are omitting is 2-qubits wide.
        '''
        if not hasancillas:
            return counts
        else:
            counts_stripped = {}
            for key,c in counts.items():
                key_stripped = key[2:].strip()
                if key_stripped in counts_stripped:
                    counts_stripped[key_stripped] += c
                else:
                    counts_stripped[key_stripped] = c
            return qiskit.result.Counts(counts_stripped)

    def get_statevectors(self):
        '''
        Retrieve list of statevectors from snapshot labeled "sv"
        '''
        return self._dict['data']['snapshots']['statevector']['sv']

    def get_rho(self):
        '''
        Retrieve density matrix from snapshot labeled "rho".
        '''
        return self._dict['data']['snapshots']['density_matrix']['rho'][0]['value']

    def get_counts(self, *_):
        return self._counts
    
    def to_dict(self):
        res = self.res.to_dict()
        for idx,_ in enumerate(res['results']):
            counts = res['results'][idx]['data'].pop('counts')
            new_counts = {}
            b_str_len = len(bin(int(max(list(counts.keys())),16))[2:])
            for k in counts:
                k_str = int(k,16) # Hex to int
                k_bin = bin(k_str)[2:] # Int to bin (minus magic bits)
                pad = '0'*(b_str_len - len(k_bin)) # Pad binary to uniform len.
                k_bin = (pad + k_bin)[2:] # Binary key, assumes 2-wide ancilla
                try:
                    new_counts[k_bin] += counts[k]
                except:
                    new_counts[k_bin] =  counts[k]
            
            res['results'][idx]['data'].update({'counts':new_counts})
        return res

def simulate_result(qc, backend, hasancillas=False, shots=1024, noise_model=None, add_delays=True, add_spam=0.0, dqpu_spec={}):
    """
    Execute simulation of circuit 'qc'.
    
    If add_delays==True, add stall & relaxation with rebuild_with_delays().
    If add_spam==True, add MCMR/SPAM errors with rebuild_mcmr_spam().
    """
    # hasancillas = False
    # for reg in qc.cregs:
    #     if 'cl_interface' in reg.name:
    #         hasancillas = True
    #         break
    
    if add_delays:
        qc_delayed = rebuild_with_delays(qc, dqpu_spec['dqpu'], dqpu_spec['delays'])
        stall_time = qc_delayed.total_stall
    else:
        qc_delayed = qc
        stall_time = 0.0

    if add_spam==0.0:
        qc_spammed = qc_delayed
    else:
        qc_spammed = rebuild_mcmr_spam(qc_delayed, prob=add_spam)

    print(f"[{time.ctime()}] Simtools: Executing circuit.")
    job = qiskit.execute(qc_spammed, backend, noise_model=noise_model, shots=shots)
    print(f"[{time.ctime()}] Simtools: Execution done.")
    
    # return result, stall_time
    return job
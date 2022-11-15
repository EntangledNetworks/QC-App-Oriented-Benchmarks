import os, sys, time, pickle

import numpy as np

from ENpackage import ENcompiler
from qiskit.compiler import transpile

# >>>>> Utility funcs to read/write
def write_data(data, file):
    try:
        handle = open(file,'wb')
        pickle.dump(data,handle)
        handle.close()
        print(f"Data written to: {file}")
    except:
        handle.close()
        raise Exception("Error writing")
    
def read_data(file):
    try:
        handle = open(file,'rb')
        data = pickle.load(handle)
        handle.close()
    except:
        handle.close()
        raise Exception("Error writing")
    
    return data
# <<<<< End of read/write func defs.

# >>>>> Script input parameters
homeDir = '/home/edrazor'
aq_repository = homeDir + '/Repos/remote/QC-App-Oriented-Benchmarks/'
workingDir = homeDir + '/Documents/data/AQ/testruns/'
num_qubits_per_qpu = 10
basis_gates = ['rx','ry','rz','cx']
gate_times = { # Gate times in seconds
    'rx'    :135e-6,
    'ry'    :135e-6,
    'rz'    :135e-6,
    'h'     :135e-6,
    'cx'    :600e-6
}
delays = { # Other timing info
    'gtime' : gate_times,
    'ictime': 1.0e-3, # Interconnect EPR generation time
    't1'    : 10.0, # T1 in seconds 
    't2'    : 1.0 # T2 in seconds
}
noise_kwargs = { # Args. for noise generators.
    'p1qb': 0.9995, # 1-qb gate fidelity
    'p2qb': 0.996, # 2-qb gate fidelity
    'p_epr': 0.99, # EPR state fidelity
    'p_spam': 0.0039 # SPAM read-out error probability
}
# <<<<< End of script input parameters


# >>>>> Setup:
os.chdir(aq_repository)
sys.path.insert(1, aq_repository+'_common/qiskit')
import simtools
import noise
sys.path.insert(1, aq_repository+'_common/')
import metrics

# Setup EN compiler and define distributed arch.
qpus = [ENcompiler.qlink.QPU(list(range(num_qubits_per_qpu + 1))), ENcompiler.qlink.QPU(list(range(num_qubits_per_qpu+1,2 * num_qubits_per_qpu + 2)))]
ic = ENcompiler.qlink.Interconnect(ENcompiler.qlink.Link(qpus[0].qubits[-1], qpus[1].qubits[-1]))
dqpu = ENcompiler.qlink.DistributedQPU(ic, *qpus)

# Instantiate noise model.
nm_dqpu, nm_mono = noise.get_noise_custom(dqpu, noise.c_noise_1qb, noise.c_noise_2qb, **noise_kwargs) # Custom (dephasing model).
# nm_dqpu, nm_mono = simtools.get_noise(dqpu, **noise_kwargs) # Get noise models. # Basic (depolarising) model.
if num_qubits_per_qpu==0: # If we are simulating a monolithic arch.
    nm = nm_mono
else: # If we are simulating a distributed arch.
    nm = nm_dqpu


def transformer_en(qc, backend=None):
    '''
    Custom transpilation routine passed to AQ's execute.py.
    '''
    basis_gates = ['rx','ry','rz','cx']
    print(f"[{time.ctime()}] Transformer: Starting QISkit transpile -O3 pass.")
    qc2 = transpile(qc, basis_gates=basis_gates, optimization_level=3)
    print(f"    O3-compiled stats| Depth: {qc2.depth()}, 2-qb gate: {qc2.count_ops()['cx']}")
    print(f"[{time.ctime()}] Transformer: Starting EN compiler pass.")
    qc3, comp = ENcompiler.transpile(qc, dqpu, hints={'decompose_remote_gates':True})
    print(f"[{time.ctime()}] Transformer: Routines completed.")

    return qc3

backend_id="aer_simulator"
hub="ibm-q"; group="open"; project="main"
provider_backend = None
exec_options = {
    'noise_model'   : nm,
    'transformer'   : transformer_en,
    'executor'      : simtools.simulate_result,
    'executor_args' : {
        'hasancillas'   : True,
        'add_delays'    : False,
        'add_spam'      : 0.0,
        'dqpu_spec'     : {'dqpu':dqpu, 'delays':delays}
    },
    'post-processor': simtools.ResultMock,
    'post-processor_args': {
        'hasancillas'   : True
    }
}
# <<<<< End of setup.

# >>>>> Define which test family / instance sizes to run.
run_specs = {
    'QAE': {
        6: [1,6,1],
        10: None,
        15: None,
        'rand': True,
        'path': 'amplitude-estimation',
        'module_name': 'ae_benchmark'
    },
    'MC': {
        6: [1,6,1],
        10: [1,3,1],
        15: None,
        'rand': True,
        'path': 'monte-carlo',
        'module_name': 'mc_benchmark'
    },
    'VQE': {
        6: [2, 6, 2],
        10: [2, 3, 2],
        15: None,
        'rand': False,
        'path': 'vqe',
        'module_name': 'vqe_benchmark'
    },
    'Shor': {
        # 6: [10,11,4],
        6: [4, 5, 4],
        # 10: [14, 19, 4],
        10: [4, 9, 4],
        # 15: [18, 27, 4], # Larger possible; practically limited 10-26, in steps of 4.
        15: [3, 12, 4],
        'rand': True,
        'path': 'shors',
        'module_name': 'shors_benchmark'
    },
    'Quantum Fourier transform': {
        6: [1,6,1],
        10: [1,10,1],
        15: [1,15,1],
        'rand': True,
        'path': 'quantum-fourier-transform',
        'module_name': 'qft_benchmark'
    },
    'Phase estimation': {
        6: [1,6,1],
        10: [1,10,1],
        15: [1,15,1],
        'rand': True,
        'path': 'phase-estimation',
        'module_name': 'pe_benchmark'
    },
    'BV': {
        6: [1,6,1],
        10: [1,10,1],
        15: [1,15,1],
        'rand': True,
        'path': 'bernstein-vazirani',
        'module_name': 'bv_benchmark'
    },
    'Hamiltonian simulation': {
        6: [1,6,1],
        10: [1,10,1],
        15: [1,6,1],
        'rand': False,
        'path': 'hamiltonian-simulation',
        'module_name': 'hamiltonian_simulation_benchmark'
    }
}

for key in run_specs: # Add paths, import modules.
    sys.path.insert(1, aq_repository+run_specs[key]['path']+'/qiskit')
    exec('import ' + run_specs[key]['module_name'] + ' as bench_mod')
    run_specs[key].update({'module': bench_mod})
    del bench_mod
# <<<<< End of definition of circuits to run.

#============#
# START RUN! #
#============#
os.chdir(workingDir)
num_shots = 4096
aq_bound = 23
for app in run_specs.keys():
    range_info = run_specs[app][num_qubits_per_qpu]
    if range_info!=None:
        low, high, step = range_info
        low += num_qubits_per_qpu
        high += num_qubits_per_qpu
        if run_specs[app]['rand']:
            max_circuits=3
        else:
            max_circuits=1

        print(f"==========Starting execution of {app}==========")
        for nqb in range(low, high, step):
            if nqb>aq_bound:
                break

            bench_mod = run_specs[app]['module']
            bench_mod.run(min_qubits=nqb, max_qubits=nqb, max_circuits=max_circuits,\
                num_shots=num_shots, backend_id=backend_id, provider_backend=provider_backend,\
                hub=hub, group=group, project=project, exec_options=exec_options)
            print("Calculated fidelity is: ", bench_mod.metrics.group_metrics['avg_fidelities'])
            isPassed, aq_est = simtools.check_aq(bench_mod.metrics, num_shots)

            try:
                fname = app.replace(' ','')+'-'+str(nqb)+'qb.dat'
                write_data([bench_mod.metrics.group_metrics,bench_mod.metrics.circuit_metrics], fname)
            finally:
                del fname
                del bench_mod
            
            if not isPassed:
                aq_bound = min(aq_est,aq_bound)
                print(f"-----Halting, fid. too low. AQ bound reset to: {aq_bound}-----")
                break
        print(f"==========Execution of {app} ended==========")
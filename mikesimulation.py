import os
import subprocess
import multiprocessing
import time
from numba import cuda

def is_cuda_available():
    """Check if CUDA is available."""
    return cuda.is_available()

def run_model(simfile_name, use_gpu=True, verbose=False):
    """Run the simulation model.

    Parameters:
    simfile_name (str): The path to the simulation file.
    use_gpu (bool): Flag to determine if GPU should be used. Defaults to True.
    verbose (bool): Flag to control print statements. Defaults to True.
    """
    # Define the paths and source scripts
    bin_path = "/teamspace/studios/this_studio/MIKE/2024/bin"
    mikevars_script = "/teamspace/studios/this_studio/MIKE/2024/mikevars.sh"
    mpi_env_script = "/teamspace/studios/this_studio/intel/oneapi/mpi/2021.7.0/env/vars.sh"

    # Update the PATH environment variable
    os.environ["PATH"] = f"{bin_path}:{os.environ['PATH']}"

    # Source the environment setup scripts using bash
    source_command = f"source {mikevars_script} && source {mpi_env_script}"

    if use_gpu:
        if verbose:
            print("Running simulation on GPU")
        command = f"FemEngineHDGPU {simfile_name}"
    else:
        if verbose:
            print("Running simulation on CPU")
        # Get the number of CPUs/cores
        num_cpus = multiprocessing.cpu_count()
        if verbose:
            print(f"Number of CPUs/cores available: {num_cpus}")
        number_of_processes = num_cpus - 1  # Use all but one CPU
        # Build the command as a list
        command = f"mpirun -n {number_of_processes} FemEngineHD {simfile_name}"

    # Combine the source commands and the main command
    full_command = f"bash -c '{source_command} && {command}'"

    # Run the command with shell=True
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if verbose:
        # Debugging output
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())

    return stdout.decode(), stderr.decode()

def execute_simulation(simfile_name, verbose=True):
    """Execute the simulation and print the execution time.

    Parameters:
    simfile_name (str): The path to the simulation file.
    verbose (bool): Flag to control print statements. Defaults to True.
    """
    use_gpu = is_cuda_available()
    if verbose:
        if use_gpu:
            print("CUDA is available")
        else:
            print("CUDA is not available")

    # run and time simulation
    start_time = time.time()
    run_model(simfile_name, use_gpu, verbose)
    end_time = time.time()
    execution_time = end_time - start_time
    if verbose:
        print(f"Execution time: {execution_time:.2f} seconds")

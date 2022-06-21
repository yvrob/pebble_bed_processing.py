import os
from attr import field

from psutil import sensors_battery

def rec_Serpent_file_search(main_input, verbose=True, level=0, log_delim='\t'):
    if verbose:
        print(f'{log_delim*level}Reading: {main_input}')    
    files_to_read = [main_input]

    path = os.path.dirname(main_input)
    with open(main_input) as f:
        lines = f.readlines()
    for i_line in range(len(lines)):
        line = lines[i_line].strip().split('%')[0]
        if len(line) == 0:
            continue
        fields = line.split()
        cmd = fields[0]
        if cmd == 'include':
            field_spot = 1
        elif cmd == 'pbed':
            field_spot = 3
        else:
            continue
        
        if isinstance(field_spot, int):
            field_spot = [field_spot]

        if len(fields) > max(field_spot):
            for i in field_spot:
                file_name = fields[i].replace('"', '')
                files_to_read += rec_Serpent_file_search(os.path.normpath(os.path.join(path, file_name)), verbose=verbose, level=level+1, log_delim=log_delim)
        else:
            spots = list(fields)
            while len(spots) <= max(field_spot):
                i_line += 1
                line = lines[i_line].strip().split('%')[0]
                while len(line) == 0:
                    i_line += 1
                    line = lines[i_line].strip().split('%')[0]
                    if i_line > len(lines):
                        raise Exception('Lost here') 
                fields = line.split()
                for j in fields:
                    spots.append(j)

            for j in field_spot:
                file_name = spots[j].replace('"', '')
                files_to_read += rec_Serpent_file_search(os.path.normpath(os.path.join(path, file_name)), verbose=verbose, level=level+1, log_delim=log_delim)
    if len(files_to_read) > 1:
        if verbose:
            print(f'{log_delim*(level+1)}Additional files ({len(files_to_read)-1}): {files_to_read[1:]}')

    return files_to_read

def analyze_sbatch_file(path_sbatch_file):
    with open(path_sbatch_file) as f:
        lines = f.readlines()
    
    ncores = ''
    for line in lines:
        fields = line.strip().split()
        if len(fields) == 0:
            continue

        if fields[0] == 'mpirun':
            for i in range(1, len(fields)):
                if fields[i] == '-np': 
                    nnodes = fields[i+1]
                elif fields[i] == '-omp':
                    ncores = fields[i+1] 
        if fields[0] == '#SBATCH' and '--ntasks=' in fields[1]:
            ntasks = int(fields[1].split('--ntasks=')[-1])
        if fields[0] == '#SBATCH' and '--cpus-per-task=' in fields[1]:
            cpus_per_task = int(fields[1].split('--cpus-per-task=')[-1])

    # Replace symbols
    if nnodes == '$SLURM_JOB_NUM_NODES':
        nnodes = ntasks
    else:
        nnodes = int(nnodes)

    if ncores == '' or ncores == '$SLURM_CPUS_PER_TASK':
        ncores = cpus_per_task
    else:
        ncores = int(ncores)
    
    return nnodes, ncores
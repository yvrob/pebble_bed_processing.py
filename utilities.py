import os

def rec_Serpent_file_search(main_input, verbose=True, level=0, log_delim='\t'):
    if verbose:
        print(f'{log_delim*level}Reading: {main_input}')
    files_to_read = [main_input]
    if main_input.split('.')[-1] == 'stl':
        return files_to_read

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
        elif cmd == 'file':
            field_spot = 2
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

def reset_case():
  path_folder_list = ['./Plots/', './Data/', './Waste/', './wrk_Serpent/']
  for path_folder in path_folder_list:
    if os.path.exists(path_folder):
      shutil.rmtree(path_folder)
    os.makedirs(path_folder)

def erase_working_plots():
  for file in glob('./wrk_Serpent/*.png'):
    os.remove(file)

def get_transferrable(transferrable, serpent_instance=None, input_parameter=False):
  if isinstance(transferrable, str):
    if transferrable[:3] != 'sss':
      if input_parameter:
        transferrable = f'sss_iv_{transferrable}'
      else:
        transferrable = f'sss_ov_{transferrable}'
    tra = serpent_instance.get_transferrable(transferrable)
  else:
    tra = transferrable
  return tra

def Serpent_get(transferrable, serpent_instance=None, input_parameter=False):
  tra = get_transferrable(transferrable, serpent_instance, input_parameter=input_parameter)
  tra.communicate()
  return tra

def Serpent_get_values(transferrable, serpent_instance=None, input_parameter=False, return_singles=True, get_errors=False):
  tra = Serpent_get(transferrable, serpent_instance, input_parameter=input_parameter)
  print(f'Getting transferrable values for "{tra.name}"')
  if not get_errors:
    values = tra.value_vec
  else:
    values = tra.uncertainty_vec
  if return_singles and len(values) == 1:
    values = values[0]
  return values

def Serpent_get_material_wise(parent_name, parameter, serpent_instance, Nmat=0, prefix='material', input_parameter=False):
  if Nmat==0:
    Nmat = sum([f'{parent_name}z' in name for name in Serpent_get_values('materials', serpent)])
  transferrables = [get_transferrable(f'{prefix}_{material_name}z{i+1}_{parameter}', serpent_instance, input_parameter=input_parameter) for i in range(Nmat)]
  return np.array(transferrables)

def Serpent_set_values(transferrable, values, serpent_instance=None, communicate=True):
  tra = get_transferrable(transferrable, serpent_instance, input_parameter=True)
  print(f'Setting transferrable "{tra.name}" to values : {values}')
  if isinstance(values, (int, float)):
    values = [values]
  tra.value_vec = np.array(values)
  if communicate:
    tra.communicate()
  return tra

def Serpent_set_multiple_values(list_transferrables, values, serpent_instance=None, communicate=True):
  if isinstance(values, (int, float)):
    values = np.array([values], dtype=type(values))
  for transferrable in list_transferrables:
    tra = get_transferrable(transferrable, serpent_instance, input_parameter=True)
    print(f'Setting transferrable "{tra.name}" to values : {values}')
    tra.value_vec = values
    if communicate:
      tra.communicate()
  return list_transferrables

def Serpent_set_option(transferrable, serpent_instance=None, turn_on=True, communicate=True):
  tra = get_transferrable(transferrable, serpent_instance, input_parameter=True)
  if turn_on:
    tra.value_vec[0] = 1
  else:
    tra.value_vec[0] = 0
    print(f'Setting transferrable "{tra.name}" to value : {tra.value_vec[0]}')
  if communicate:
    tra.communicate()
  return tra

def assign_random_array(source_list, possible_values):
  shuffled_list = np.array_split(np.random.permutation(source_list), len(possible_values))
  array = np.empty(Nfuel, dtype=int)
  for i, val in enumerate(possible_values):
    for j in range(len(shuffled_list[i])):
      array[shuffled_list[i][j]] = val
  return array

def get_index_table(list1, list2):
  index_list = np.ones(len(list1))*np.nan
  for j in range(len(list1)):
    for k in range(len(list2)):
        if list1[j] == list2[k]:
          index_list[j] = k
          break
  return index_list

def estimate_burnup(z, zlim, direction, pass_number, max_passes, max_bu, mult):
  if direction == -1:
    bu = (zlim[1] - z)/(zlim[1] - zlim[0])  *(pass_number/max_passes) * max_bu * mult
  elif direction == +1:
    bu = (z - zlim[0])/(zlim[1] - zlim[0])  *(pass_number/max_passes) * max_bu * mult
  return bu

def load_interpolator(interpolator_path, to_replace=[666], replace_value=[-1]):
  with open(interpolator_path, 'rb') as f:
    interpolator, zai_interpolator = pickle.load(f)
  zai_interpolator = np.array(zai_interpolator).astype(int)
  zai_interpolator[zai_interpolator == 666] = -1
  return interpolator, zai_interpolator

def interpolate_adens(interpolator, bu, index_table):
  adens_interpolated = interpolator(bu)
  adens = np.ones(len(zai_list))*np.nan
  for j in range(len(index_list)):
    if not np.isnan(index_list[j]):
      adens[j] = adens_interpolated[int(index_table[j])]
    else:
      adens[j] = 0.0
  return adens
import serpentTools
import pandas as pd
import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits import mplot3d
from copy import deepcopy
from glob import glob
import matplotlib.image as mpimg
import os

#%matplotlib widget

class Pebble_bed:
    
    #### GENERAL ####

    def __init__(self, verbose=True, level=0, log_delim='  '):
        log_print(f'Creating empty Pebble_bed object', verbose, level, log_delim)
        self.read_files = []

    def __repr__(self) -> str:
        if hasattr(self, 'data'):
            return self.data.__repr__()
        else:
            return 'Empty Pebble_bed object'

    #### READING ####

    def read_file(self, pbed_file_path, *, calculate_dist=True, calculate_radial_dist=True, radial_center=[0,0], verbose=True, level=0, log_delim='  '):
        log_print(f'Reading pbed file from {pbed_file_path}', verbose, level, log_delim)
        data = pd.read_csv(pbed_file_path, delim_whitespace=True, header=None, names=["x", "y", "z", "r", "uni"])       
        self.pbed_path = pbed_file_path
        self.data = data
        self.data['id'] = np.array(self.data.index)
        self.process_data(calculate_dist, calculate_radial_dist, radial_center, verbose=verbose, level=level+1, log_delim=log_delim)
        self.read_files.append(pbed_file_path)
        log_print(f'Done.', verbose, level, log_delim, end_block=True)
        return data    

    def read_table(self, data, universes_included=False, calculate_dist=True, calculate_radial_dist=True, radial_center=[0,0], verbose=True, level=0, log_delim='  '):
        log_print(f'Generating data from table containing xyzr', verbose, level, log_delim)
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            if universes_included:
                self.data = pd.DataFrame(data, columns=['x', 'y', 'z', 'r', 'uni'])
            else:
                self.data = pd.DataFrame(data, columns=['x', 'y', 'z', 'r'])
        self.data['id'] = np.array(self.data.index)
        self.process_data(calculate_dist, calculate_radial_dist, radial_center, verbose=verbose, level=level+1, log_delim=log_delim)
        log_print(f'Done.', verbose, level, log_delim, end_block=True)


    def read_xyzr(self, sss_ov_xyzr, calculate_dist=True, calculate_radial_dist=True, radial_center=[0,0], verbose=True, level=0, log_delim='  '):
        log_print(f'Generating data from Serpent sss_ov_xyzr field', verbose, level, log_delim)
        data = np.reshape(sss_ov_xyzr, (-1, 4))
        self.data = pd.DataFrame(data, columns=['x', 'y', 'z', 'r'])
        self.data['id'] = np.array(self.data.index)
        self.process_data(calculate_dist, calculate_radial_dist, radial_center, verbose=verbose, level=level+1, log_delim=log_delim)
        log_print(f'Done.', verbose, level, log_delim, end_block=True)

    #### PROCESSING RAW DATA ####

    def process_data(self, calculate_dist=True, calculate_radial_dist=True, calculate_angles=True, calculate_center=True, calculate_ghosts=True, radial_center=[0,0], verbose=True, level=0, log_delim='  '):
        log_print(f'Processing pbed data', verbose, level, log_delim)
        if calculate_ghosts:
            self.filter_ghosts()
        if calculate_center:
            self.center = np.array(radial_center+[np.nanmean(self.filter_ghosts().data.z)])
        if calculate_dist:
            self.data["dist"] = np.linalg.norm(self.data[["x", "y", "z"]] - self.center, axis=1)
        if calculate_radial_dist:
            self.data["r_dist"] = np.linalg.norm(self.data[["x", "y"]] - self.center[:2], axis=1)
        if calculate_angles:
            self.data["azim_angle"] = np.arctan2(-(self.data.y-self.center[0]), -(self.data.x-self.center[1]))*180/np.pi+180 # made to match Serpent angles (y=0 is reference, reversed clock)
        self.N_elements = len(self.data)
        log_print(f'Summary:', verbose, level+1, log_delim)
        log_print(f'Number of elements: {self.N_elements}', verbose, level+2, log_delim)
        log_print(f'Fields: {list(self.data.columns)}', verbose, level+2, log_delim)
        self.radii_list = self.data.r.unique()
        log_print(f'Radii: {self.radii_list}', verbose, level+2, log_delim)
        self.box = np.array([[self.data.x.min(), self.data.x.max()],
                             [self.data.y.min(), self.data.y.max()],
                             [self.data.z.min(), self.data.z.max()]])
        log_print(f'[x,y,z] limits: [{self.box[0,0]:.2f}, {self.box[0,1]:.2f}], [{self.box[1,0]:.2f}, {self.box[1,1]:.2f}],[{self.box[2,0]:.2f}, {self.box[2,1]:.2f}]', verbose, level+2, log_delim)

        if 'uni' in self.data.columns:
            self.universes_list = self.data.uni.unique()
            log_print(f'Universes: {list(self.universes_list)}', verbose, level+2, log_delim)
        log_print(f'Done.', verbose, level, log_delim, end_block=True)

    def filter_ghosts(self, filter=True):
        if filter:
            subpbed = deepcopy(self)
            are_ghosts = self.is_ghost()
            self.N_ghosts = sum(are_ghosts)
            subpbed.data.loc[are_ghosts, :] = np.nan
        return subpbed

    def is_ghost(self, index='all'):
        if isinstance(index, str) and index=='all':
            return np.logical_or(self.data['r'] == 0, np.isnan(self.data['r']))
        else:    
            return np.logical_or(self.data.loc[index, 'r'] == 0, np.isnan(self.data.loc[index, 'r']))

    #### PROCESSING DATA ####

    def to_xyzr(self):
        return np.array(self.data[['x','y','z','r']]).flatten()

    def extract(self, fields='all', indices='all'):
        subpbed = deepcopy(self)

        if isinstance(fields, (list, tuple, np.ndarray)) or (isinstance(fields, str) and fields != 'all'):
            subpbed.data = subpbed.data[fields]

        if isinstance(indices, (list, tuple, np.ndarray)) or isinstance(indices, int):
           subpbed.data = subpbed.data.loc[indices]
        
        return subpbed

    #### POST PROCESSING ####

    def add_field(self, name, array, err=False, verbose=True, level=0, log_delim='  '):
        log_print(f'Adding field {name} manually to data (N={len(array)})', verbose, level, log_delim)
        if not hasattr(self, 'fields'):
            self.fields = dict()
        if not err:
            self.fields[name] = array
            if len(array) == len(self.data):
                self.data[name] = self.fields[name]
        else:
            self.fields_err[name] = array
            if len(array) == len(self.data):
                self.data[name+'_err'] = self.fields_err[name]  
        log_print(f'Done.', verbose, level, log_delim, end_block=True)

    def read_detector(self, det_file_path, which_dets='all', verbose=True, level=0, log_delim='  '):
        log_print(f'Reading det file from {det_file_path}', verbose, level, log_delim)
        det_reader = serpentTools.read(det_file_path, reader='det')
        if not hasattr(self, 'detector_names'):
            self.detector_names = []
        if not hasattr(self, 'fields'):
            self.fields = dict()
        if not hasattr(self, 'fields_err'):
            self.fields_err = dict()
        if not hasattr(self, 'fields_grids'):
            self.fields_grids = dict()

        if which_dets=='all':
            which_dets = list(det_reader.detectors.keys())

        for det_name in which_dets:
            tallies = det_reader[det_name].tallies
            errors = det_reader[det_name].errors
            grids = det_reader[det_name].grids
            self.detector_names.append(det_name)
            self.fields[det_name] = tallies
            self.fields_err[det_name] = errors
            self.fields_grids[det_name] = grids

            # Handles up to 1 extra grid level
            if np.atleast_2d(tallies).shape[1] == len(self.data) and len(grids) <=1:
                if len(grids) == 0:
                    self.data[det_name] = self.fields[det_name]
                    self.data[det_name + '_err'] = self.fields_err[det_name]
                else:
                    name_grid = list(grids)[0]
                    for i_grid in range(grids[name_grid].shape[0]):
                        self.data[f'{det_name}_{name_grid}{i_grid}'] = self.fields[det_name][i_grid, :]
                        self.data[f'{det_name}_{name_grid}{i_grid}_err'] = self.fields_err[det_name][i_grid, :]                        
        self.read_files.append(det_file_path)
        log_print(f'Added following detectors: {which_dets}', verbose, level+1, log_delim)
        log_print(f'Done.', verbose, level, log_delim, end_block=True)

    def read_depletion(self, dep_file_path, material_name, dd=False, steps='all', fields='all', verbose=True, level=0, log_delim='  '):
        if dd:
            if dep_file_path[-2:] == '.m':
                dep_file_path = dep_file_path[:-2]
            dep_file_path = glob(dep_file_path + '_dd*.m')
        else:
            if dep_file_path[-2:] != '.m':
                dep_file_path += '.m'
            dep_file_path = [dep_file_path]
        

        log_print(f'Reading dep file from {dep_file_path}', verbose, level, log_delim)
        dep_reader = serpentTools.read(dep_file_path[0], reader='dep')
        for i in range(1, len(dep_file_path)):
            additional_reader = serpentTools.read(dep_file_path[i], reader='dep')
            for mat_name, mat in additional_reader.materials.items():
                dep_reader.materials[mat_name] = mat

        if not hasattr(self, 'fields'):
            self.fields = dict()
            
        if fields=='all':
            fields = [i for i in list(dep_reader.materials.values())[0].data.keys()]
        if steps=='all':
            steps = [i for i in range(len(list(dep_reader.materials.values())[0].burnup))]
        if material_name=='infer':
            candidates = []
            for name in [mat.name for mat in dep_reader.materials.values()]:
                if 'z' in name and name.split('z')[-1].isdigit():
                    candidates.append(name)
            if len(candidates) == 0:
                raise Exception('No divided material found')
            if len(candidates) > 1:
                raise Exception('More than one divided material found, please choose between: ', candidates)
            else:
                material_name = candidates[0]

        nuc_wise = []
        nuc_names = dep_reader.materials[f'{material_name}z1'].names
        for step in steps:
            for field_name in fields:
                if isinstance(dep_reader.materials[f'{material_name}z1'].data[field_name][0], (tuple, list, np.ndarray)):
                    for name in nuc_names:
                        self.fields[f'{name}_{field_name}_{step}'] = np.empty(len(self.data))
                    nuc_wise.append(True)      
                else:
                    self.fields[f'{field_name}_{step}'] = np.empty(len(self.data))      
                    nuc_wise.append(False)
        for step in steps:
            for i in range(1, len(self.data)+1):
                for i_field, field_name in enumerate(fields):
                    if nuc_wise[i_field]:
                        for i_name, name in enumerate(nuc_names):
                            self.fields[f'{name}_{field_name}_{step}'][i-1] = dep_reader.materials[f'{material_name}z{i}'].data[field_name][step][i_name]
                    else:
                        self.fields[f'{field_name}_{step}'][i-1] = dep_reader.materials[f'{material_name}z{i}'].data[field_name][step]  

            for i_field, field_name in enumerate(fields):
                if nuc_wise[i_field]:
                    for name in nuc_names:
                        self.data[f'{name}_{field_name}_{step}'] = self.fields[f'{name}_{field_name}_{step}']
                else:
                    self.data[f'{field_name}_{step}'] = self.fields[f'{field_name}_{step}']

        for file in dep_file_path:
            self.read_files.append(file)

        log_print(f'Added following fields for step(s) {steps}: {fields}', verbose, level+1, log_delim)
        log_print(f'Done.', verbose, level, log_delim, end_block=True)

    #### SORTING DATA ####

    def slice(self, dir_id=0, val='middle', *, tol=None, verbose=True, level=0, log_delim='  '):
        dir_id = str(dir_id)
        if dir_id.lower() in ["x", "y", "z"]:
            dir_id = ["x", "y", "z"].index(dir_id.lower())
        dir_id = int(dir_id)

        if isinstance(val, str) and val == 'middle':
            val = np.nanmean(self.filter_ghosts().data[['x','y','z'][dir_id]])

        log_print(f'Slicing pbed in direction {dir_id} at value {val:.4E}', verbose, level, log_delim)
        if isinstance(tol, type(None)):
            log_print('Tolerance set to radii values', verbose, level=level+1, log_delim=log_delim)
        else:
            log_print(f'Tolerance set to {tol}', verbose, level=level+1, log_delim=log_delim)

        sub_pbed = deepcopy(self)

        if isinstance(tol, type(None)):
            sub_pbed.data = self.data[np.abs(self.data[["x", "y", "z"][dir_id]] - val) <= self.data["r"]]
        else:
            sub_pbed.data =  self.data[np.abs(self.data[["x", "y", "z"][dir_id]] - val) <= tol]
        sub_pbed.N_elements

        log_print(f'Done.', verbose, level, log_delim, end_block=True)
        return sub_pbed

    def clip(self, dir_id=0, val='middle', direction=+1, verbose=True, level=0, log_delim='  '):
        sdir = '>=' if direction==1 else '<='
        sub_pbed = deepcopy(self)
        dir_id = str(dir_id)

        if isinstance(val, str) and val == 'middle':
            val = np.nanmean(self.filter_ghosts().data[['x','y','z'][dir_id]])

        log_print(f'Clipping pbed in direction {dir_id} at values {sdir} {val:.4E}', verbose, level, log_delim)

        if dir_id.lower() in ["x", "y", "z"]:
            dir_id = ["x", "y", "z"].index(dir_id.lower())
            sub_pbed.data =  self.data[self.data[["x", "y", "z"][dir_id]]*direction >= val*direction]
        elif dir_id == '4' or dir_id.lower() in ['dist', 'd']:
            sub_pbed.data =   self.data[self.data.dist*direction >= val*direction]
        elif dir_id == '5' or dir_id.lower() in ['r_dist', 'rdist', 'r']:
            sub_pbed.data = self.data[self.data.r_dist*direction >= val*direction]  
        else:
            sub_pbed.data =  self.data[self.data[["x", "y", "z"][dir_id]]*direction >= val*direction]

        log_print(f'Done.', verbose, level, log_delim, end_block=True)
        return sub_pbed

    #### PLOTTING ####

    def projection(self, dir_id, val, verbose=True, level=0, log_delim='  '):
        dir_id = str(dir_id)
        if dir_id.lower() in ["x", "y", "z"]:
            dir_id = ["x", "y", "z"].index(dir_id.lower())
        dir_id = int(dir_id)
        
        log_print(f'Projecting pbed in direction {dir_id} at value {val:.4E}', verbose, level, log_delim)
        rel_pos = val - self.data[["x", "y", "z"][dir_id]]
        tmp = self.data.r**2 - rel_pos**2
        r_projected = np.ones_like(tmp)*np.nan
        r_projected[tmp >= 0] = tmp[tmp >= 0]**0.5
        
        log_print(f'Done.', verbose, level, log_delim, end_block=True)
        return r_projected

    def plot2D(self, field='id', dir_id=0, val='middle', colormap='turbo', xlim=None, ylim=None, tol=None, equal=True, field_title=None, clim=None, superimpose_Serpent=False, Serpent_xsize=None, Serpent_ysize=None, Serpent_geom_path=None, fig_size=None, new_fig=True, save_fig=False, fig_folder='./', fig_name=None, fig_suffix='', fig_dpi=600, plot_title=None, verbose=True, level=0, log_delim='  '):
        dir_id = str(dir_id)
        if dir_id == '0' or dir_id.lower() == 'x':
            xdir = 1
            ydir = 2
            dir_id = 0
        elif dir_id == '1' or dir_id.lower() == 'y':
            xdir = 0
            ydir = 2
            dir_id = 1
        elif dir_id == '2' or dir_id.lower() == 'z':
            xdir = 0
            ydir = 1
            dir_id = 2
        
        if isinstance(val, str) and val == 'middle':
            val = np.nanmean(self.filter_ghosts().data[['x','y','z'][dir_id]])

        if isinstance(field, (tuple, list, np.ndarray, pd.DataFrame, pd.Series)):
            log_print(f'2D plotting pbed in direction {dir_id} at value {val:.4E}, showing user-defined array', verbose, level, log_delim)
            if isinstance(field_title, type(None)):
                self.data['user-defined'] = np.array(field)
                field = 'user-defined'
                field_title = field
            else:
                self.data[field_title] = np.array(field)
                field = field_title

        else:
            if isinstance(field_title, type(None)):
                field_title = field
            log_print(f'2D plotting pbed in direction {dir_id} at value {val:.4E}, showing field {field}', verbose, level, log_delim)
        
        dir_id = int(dir_id)        
        if isinstance(tol, type(None)):
            sub_pbed = self.slice(dir_id, val, verbose=verbose, level=level+1, log_delim=log_delim)
            r = sub_pbed.projection(dir_id, val, verbose=verbose, level=level+1, log_delim=log_delim)
        else:
            sub_pbed = self.slice(dir_id, val, verbose=verbose, tol=tol, level=level+1, log_delim=log_delim)
            r = np.array(sub_pbed.data.r)

        data = sub_pbed.data[sub_pbed.data.r != 0]

        x = np.array(data[["x", "y", "z"][xdir]])
        y = np.array(data[["x", "y", "z"][ydir]])

        patches = []
        for i in range(len(data)):
            circle = Circle((x[i], y[i]), r[i])
            patches.append(circle)
        
        colors = np.array(data[field])

        if isinstance(clim, type(None)):
            clim = [data[field].min(), data[field].max()]  

        p = PatchCollection(patches, cmap=colormap)
        p.set_array(colors)
        p.set_clim(clim)
    
        if new_fig:
            if not isinstance(fig_size, type(None)):
                plt.figure(figsize=fig_size)
            else:
                plt.figure()

        if superimpose_Serpent:
            log_print(f'Superimposing Serpent geometry (x={Serpent_xsize}, y={Serpent_ysize}) from {Serpent_geom_path}', verbose, level+1, log_delim)
            self.show_Serpent_plot(Serpent_geom_path, new_fig=False, xlim=Serpent_xsize, ylim=Serpent_ysize, verbose=verbose, level=level+1, log_delim=log_delim)

        ax = plt.gca()
        ax.add_collection(p)
        plt.colorbar(p, label=field_title, orientation="horizontal") # shrink=0.5
        if not isinstance(xlim, type(None)):
            ax.set_xlim(xlim)
        if not isinstance(ylim, type(None)):
            ax.set_ylim(ylim)
        plt.xlabel(["x", "y", "z"][xdir])
        plt.ylabel(["x", "y", "z"][ydir])
        if isinstance(plot_title, type(None)):
            plt.title("{}={:.3f}".format(["x", "y", "z"][dir_id], val))
        else:
            plt.title(plot_title)
        ax.autoscale_view()
        if equal:
            ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()

        if save_fig:
            if isinstance(fig_name, type(None)):
                fig_path = f'{fig_folder}/2D_plot_{["x", "y", "z"][dir_id]}{val:.2E}_{field}{fig_suffix}.png'
            else:
                fig_path = f'{fig_folder}/{fig_name}{fig_suffix}.png'
            log_print(f'Saving figure in {fig_path}', verbose, level+1, log_delim)
            plt.savefig(fig_path, dpi=fig_dpi, bbox_inches='tight')

        log_print(f'Done.', verbose, level, log_delim, end_block=True)
        return ax

    def plot3D(self, field='id', colormap='turbo', view=None, xlim=None, ylim=None, zlim=None, sample_fraction=None, fast=False, force_slow=False, scatter_size=10, alpha=1, show_ghosts=False, field_title=None, clim=None, equal=True, fig_size=None, new_fig=True, save_fig=False, fig_folder='./', fig_name=None, fig_suffix='', fig_dpi=600, verbose=True, level=0, log_delim='  '):
        lim_fast = 1000

        data = self.data[self.data.r != 0]

        if not isinstance(sample_fraction, type(None)):
            data = data.sample(int(sample_fraction*len(data)))

        if isinstance(field, (tuple, list, np.ndarray, pd.DataFrame, pd.Series)):
            log_print(f'3D plotting pbed, showing user-defined array', verbose, level, log_delim)
            if isinstance(field_title, type(None)):
                field_title = 'user-defined'
                data[str(field_title)] = np.array(field)
                field = field_title
            else:
                data[field_title] = np.array(field)
                field = field_title
        else:
            if isinstance(field_title, type(None)):
                field_title = field
            log_print(f'3D plotting pbed, showing field {field}', verbose, level, log_delim)

        if not isinstance(sample_fraction, type(None)):
            log_print(f'Only showing {sample_fraction*100}% ({len(data)}) of the elements', verbose, level+1, log_delim)

        if isinstance(clim, type(None)):
            clim = [data[field].min(), data[field].max()]
        if new_fig:
            if not isinstance(fig_size, type(None)):
                plt.figure(figsize=fig_size)
            else:
                plt.figure()
            plt.gcf().add_subplot(111, projection="3d")  # , proj_type = 'ortho')

        ax = plt.gca()
        if len(data) > lim_fast and not force_slow:
            log_print(f'WARNING: Too many elements ({len(data)}>{lim_fast}), switched automatically to scatter plot, needing manual parameter for scatter size', verbose, level+1, log_delim)
            fast = True
            if isinstance(scatter_size, type(None)):
                raise Exception('Since the number of elements is too high, plot3D switched to scatter. Please put a size to elements, with scatter_size')
        
        if force_slow and len(data) > lim_fast:
            log_print(f'WARNING: Many elements to plot ({len(data)}>{lim_fast}), might take a while', verbose, level+1, log_delim)

        if not fast:
            values =  data[field]
            cmap = cm.get_cmap(colormap)
            norm = Normalize(vmin = clim[0], vmax = clim[1])
            normalized_values = norm(values)
            colors = cmap(normalized_values)

            i_row = 0
            for _, row in data.iterrows():
                # draw sphere
                u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
                x = row.r*np.cos(u)*np.sin(v)
                y = row.r*np.sin(u)*np.sin(v)
                z = row.r*np.cos(v)
                p = ax.plot_surface(x+row.x, y+row.y, z+row.z, color=colors[i_row], shade=False, alpha=alpha, zorder=1)
                i_row += 1
            cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label=field_title, orientation="horizontal") # shrink=0.5
            
        else:
            if isinstance(scatter_size, type(None)):        
                raise Exception(f'Fast mode, needing manual parameter for scatter size')

            p = ax.scatter3D(
                data.x,
                data.y,
                data.z,
                s=scatter_size,
                c=data[field],
                alpha=alpha,
                zorder=1,
                cmap=colormap
            )
            cb = plt.colorbar(p, label=field_title, orientation="horizontal") # shrink=0.5
            cb.set_alpha(1)
            cb.draw_all()

        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_zlabel("z [cm]")

        if not isinstance(view, type(None)):
            ax.view_init(view[0], view[1])

        ax.relim()      # make sure all the data fits
        ax.autoscale()  # auto-scale
        if not isinstance(xlim, type(None)):
            ax.set_xlim(xlim)
        if not isinstance(ylim, type(None)):
            ax.set_ylim(ylim)
        if not isinstance(zlim, type(None)):
            ax.set_zlim([zlim[0], zlim[1]])
        #ax.invert_zaxis()
        if equal:
            ax.set_box_aspect((np.diff(ax.get_xlim())[0],np.diff(ax.get_ylim())[0],np.diff(ax.get_zlim())[0]))
        plt.tight_layout()

        if save_fig:
            if isinstance(fig_name, type(None)):
                fig_path = f'{fig_folder}/3D_plot_{field}{fig_suffix}.png'
            else:
                fig_path = f'{fig_folder}/{fig_name}{fig_suffix}.png'
            log_print(f'Saving figure in {fig_path}', verbose, level+1, log_delim)
            plt.savefig(fig_path, dpi=fig_dpi, bbox_inches='tight')
        
        log_print(f'Done.', verbose, level, log_delim, end_block=True)
        
        return ax

    def plot_summary(self, field='id', colormap='turbo', view=None, xlim=None, ylim=None, zlim=None, sample_fraction=None, fast=False, force_slow=False, scatter_size=10, alpha=1, field_title=None, clim=None, superimpose_Serpent=False, Serpent_xsize=None, Serpent_ysize=None, Serpent_zsize=None, Serpent_paths_dirxyz = [None, None, None], save_fig=False, fig_size=None, fig_folder='./', fig_name=None, fig_suffix='', fig_dpi=600, verbose=True, level=0, log_delim='  '):
        if not isinstance(fig_size, type(None)):
            fig = plt.figure(figsize=fig_size)
        else:
            fig = plt.figure()
        
        log_print(f'Plotting of Pebble Bed', verbose, level, log_delim)
        if superimpose_Serpent:
            valx = np.mean(Serpent_xsize)
            valy = np.mean(Serpent_ysize)
            valz = np.mean(Serpent_zsize)
        else:
            valx = 'middle'
            valy = 'middle'
            valz = 'middle'

        ax1=fig.add_subplot(2,2,1)
        self.plot2D(field, dir_id=0, val=valx, plot_title='YZ', superimpose_Serpent=superimpose_Serpent, Serpent_xsize=Serpent_ysize, Serpent_ysize=Serpent_zsize, Serpent_geom_path=Serpent_paths_dirxyz[0], new_fig=False, xlim=xlim, ylim=ylim, clim=clim, field_title=field_title, colormap=colormap, verbose=verbose, level=level+1, log_delim=log_delim)
        ax2=fig.add_subplot(2,2,2)
        self.plot2D(field, dir_id=1, val=valy, plot_title='XZ', superimpose_Serpent=superimpose_Serpent, Serpent_xsize=Serpent_xsize, Serpent_ysize=Serpent_zsize, Serpent_geom_path=Serpent_paths_dirxyz[1], new_fig=False, xlim=xlim, ylim=ylim, clim=clim, field_title=field_title, colormap=colormap, verbose=verbose, level=level+1, log_delim=log_delim)
        ax3=fig.add_subplot(2,2,3)
        self.plot2D(field, dir_id=2, val=valz, plot_title='XY', superimpose_Serpent=superimpose_Serpent, Serpent_xsize=Serpent_xsize, Serpent_ysize=Serpent_ysize, Serpent_geom_path=Serpent_paths_dirxyz[2], new_fig=False, xlim=xlim, ylim=ylim, clim=clim, field_title=field_title, colormap=colormap, verbose=verbose, level=level+1, log_delim=log_delim)
        ax4=fig.add_subplot(2,2,4, projection="3d")
        self.plot3D(field, new_fig=False, xlim=xlim, ylim=ylim, zlim=zlim, clim=clim, field_title=field_title, colormap=colormap, view=view, sample_fraction=sample_fraction, fast=fast, force_slow=False, scatter_size=scatter_size, alpha=alpha, verbose=verbose, level=level+1, log_delim=log_delim)
        
        if save_fig:
            if isinstance(fig_name, type(None)):
                fig_path = f'{fig_folder}/Summary_plot_{field}{fig_suffix}.png'
            else:
                fig_path = f'{fig_folder}/{fig_name}{fig_suffix}.png'
            log_print(f'Saving figure in {fig_path}', verbose, level+1, log_delim)
            plt.savefig(fig_path, dpi=fig_dpi, bbox_inches='tight')

        log_print(f'Done.', verbose, level, log_delim, end_block=True)
        return (ax1, ax2, ax3, ax4)

    def show_Serpent_plot(self, plot_file_path, title=None, new_fig=True, fig_size=None, save_fig=False, fig_folder='./', fig_name=None, fig_suffix='', fig_dpi=600, xlim=None, ylim=None, verbose=True, level=0, log_delim='  '):
        log_print(f'Showing Serpent plot from {plot_file_path}', verbose, level, log_delim)
        if isinstance(title, type(None)):
            title = plot_file_path
        
        if new_fig:
            if not isinstance(fig_size, type(None)):
                plt.figure(figsize=fig_size)
            else:
                plt.figure()

        ax = plt.gca()
        img = mpimg.imread(plot_file_path)
        if not isinstance(xlim, type(None)) and not isinstance(ylim, type(None)):
            imgplot = plt.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
        else:
            imgplot = plt.imshow(img)
        plt.title(title)
        plt.tight_layout()

        if save_fig:
            if isinstance(fig_name, type(None)):
                fig_path = f'{fig_folder}/Serpent_plot_{os.path.basename(plot_file_path)}{fig_suffix}.png'
            else:
                fig_path = f'{fig_folder}/{fig_name}{fig_suffix}.png'
            log_print(f'Saving figure in {fig_path}', verbose, level+1, log_delim)
            plt.savefig(fig_path, dpi=fig_dpi, bbox_inches='tight')

        log_print(f'Done.', verbose, level, log_delim, end_block=True)
        return ax

    #### WRITING ####

    def write_fields(self, path, fields='all', indices='all', separate_fields=False, separate_indices=False, sep ='\t', include_headers=True, include_indices=True, prefix='data', float_format='%.4E', verbose=True, level=0, log_delim='  '):
        subpbed = self.extract(fields=fields, indices=indices)
        if separate_fields:
            for field in subpbed.data.columns:
                if separate_indices:
                    for index in subpbed.data.index:
                        name = f'{path}/{prefix}_index_{index}_field_{field}.txt'
                        subpbed.data.loc[index, field].to_csv(name, header=include_headers, index=include_indices, sep=sep, float_format=float_format)
                else:
                    name = f'{path}/{prefix}_field_{field}.txt'
                    subpbed.data[field].to_csv(name, header=include_headers, index=include_indices, sep=sep, float_format=float_format)
        else:
            if separate_indices:
                for index in subpbed.data.index:
                    name = f'{path}/{prefix}_index_{index}.txt'
                    subpbed.data.loc[index].to_csv(name, header=include_headers, index=include_indices, sep=sep, float_format=float_format)
            else:
                name = f'{path}/{prefix}.txt'
                subpbed.data.to_csv(name, header=include_headers, index=include_indices, sep=sep, float_format=float_format)    

        
    #### DOMAIN DECOMPOSITION ####

    def decompose_in_domains(self, n_domains_list, decomposition_types, filling_domain=None, center=None, shift_sectors=0, keep_subdomains=True, verbose=True, level=0, log_delim='  '):
        self.data.drop(self.data.filter(like='domain_id'), axis=1, inplace=True)
        if isinstance(n_domains_list, int):
            n_domains_list = [n_domains_list]
        N_levels = len(n_domains_list)
        if isinstance(shift_sectors, (int)):
            shift_sectors = [[shift_sectors for j in range(n_domains_list[i])] for i in range(N_levels)]
        elif isinstance(shift_sectors, (tuple, list, np.ndarray)):
            shift_sectors = [shift_sectors for i in range(N_levels)]

        if isinstance(center, (int, type(None))):
            center = [[center for j in range(n_domains_list[i])] for i in range(N_levels)]
        elif isinstance(center, (tuple, list, np.ndarray)):
            center = [center for i in range(N_levels)]

        if N_levels != len(decomposition_types):
            raise Exception('List of #domains must match decomposition types in size (please use 1 letter alias)')

        # First decomposition
        self.decompose_in_domains_simple(n_domains_list[0], decomposition_types[0], filling_domain, center[0][0], shift_sectors[0][0], verbose=verbose, level=level+1, log_delim=log_delim)
        self.data['domain_id_0'] = np.array(self.data.domain_id)

        # Rest of decompositions
        for i_decomp in range(1, N_levels):
            n_domains = n_domains_list[i_decomp]
            level_domains = n_domains_list[i_decomp-1]
            decomposition_type = decomposition_types[i_decomp]
            domain_id = np.ones(len(self.data))*-1
            for i_domain in range(level_domains):
                shift_sector = shift_sectors[i_decomp][i_domain]
                ctr = center[i_decomp][i_domain]
                sub_pbed = deepcopy(self)
                sub_pbed.data = sub_pbed.data[sub_pbed.data[f'domain_id_{i_decomp-1}'] == i_domain]
                sub_pbed.data = sub_pbed.data.reset_index(drop=True)
                sub_pbed.decompose_in_domains_simple(n_domains, decomposition_type, filling_domain, ctr, shift_sector, idle=False, verbose=verbose, level=level+1, log_delim=log_delim)
                domain_id[list(sub_pbed.data.id)] = sub_pbed.data.domain_id
            self.data[f'domain_id_{i_decomp}'] = np.array(domain_id)
            
        self.data = self.data.sort_values([f'domain_id_{i_decomp}' for i_decomp in range(N_levels)])
        self.data.domain_id = self.data.set_index([f'domain_id_{i_decomp}' for i_decomp in range(N_levels)]).index.factorize()[0]
        
        self.data = self.data.sort_values('id')
        are_ghosts = self.is_ghost()
        self.data.loc[are_ghosts, 'domain_id'] = np.nan
        self.domains = dict()
        _, self.cnt_domains = np.unique(self.data.domain_id, return_counts=True)
        for i_domain in range(int(self.data.domain_id.max())+1):
            self.domains[i_domain] = self.cnt_domains[i_domain]
        log_print(f'Count in domains: {self.cnt_domains}', verbose, level+1, log_delim)

        if not keep_subdomains:
            self.data.drop(self.data.filter(like='domain_id_'), axis=1, inplace=True)

    def decompose_in_domains_simple(self, n_domains, decomposition_type, filling_domain=None, center=None, shift_sector=0, idle=True, verbose=True, level=0, log_delim='  '):

        if isinstance(filling_domain, type(None)):
            filling_domain = n_domains-1
        
        if not isinstance(center, type(None)):
            self.center = center
            self.process_data(calculate_center=False, verbose=verbose, level=level+1, log_delim=log_delim)
        else:
            self.process_data(verbose=verbose, level=level+1, log_delim=log_delim)

        decomposition_type = str(decomposition_type).lower()
        target_npebbles_domains = np.floor((len(self.data)- self.N_ghosts)/ n_domains).astype(int)

        if decomposition_type in ['0', 'random', 'n']:
            
            log_print(f'Decomposing pebbles randomly in {n_domains} domains', verbose, level, log_delim)
            domain_id = []
            i_pebble = 0
            are_ghosts = self.is_ghost()
            for i_domain in range(n_domains):
                cnt_domain = 0 
                while cnt_domain < target_npebbles_domains:
                    if not are_ghosts[i_pebble]:
                        domain_id.append(i_domain)
                        cnt_domain += 1
                    else:
                        domain_id.append(np.nan)
                    i_pebble += 1
            
            # Handles last domain
            cnt_extra = 0
            while len(domain_id) < len(self.data):
                domain_id.append(filling_domain)
                cnt_extra += 1
            if cnt_extra > 0:
                log_print(f'{cnt_extra} extra pebbles in domain {filling_domain}', verbose, level+1, log_delim)
            
            # Shuffles
            np.random.shuffle(domain_id)

        elif decomposition_type in ['1', 'index', 'i', 'id']:
            
            log_print(f'Decomposing pebbles by index in {n_domains} domains', verbose, level, log_delim)
            domain_id = []
            i_pebble = 0
            are_ghosts = self.is_ghost()
            for i_domain in range(n_domains):
                cnt_domain = 0 
                while cnt_domain < target_npebbles_domains:
                    if not are_ghosts[i_pebble]:
                        domain_id.append(i_domain)
                        cnt_domain += 1
                    else:
                        domain_id.append(np.nan)
                    i_pebble += 1

            # Handles last domain
            cnt_extra = 0
            while len(domain_id) < len(self.data):
                domain_id.append(filling_domain)
                cnt_extra += 1
            if cnt_extra > 0:
                log_print(f'{cnt_extra} extra pebbles in domain {filling_domain}', verbose, level+1, log_delim)
            
        elif decomposition_type in ['2', 'sectors', 'sector', 's']:
            
            log_print(f'Decomposing pebbles by azimuthal sectors in {n_domains} domains', verbose, level, log_delim)
            if shift_sector > 0:
                log_print(f'Shifting sectors by {shift_sector} degree', verbose, level+1, log_delim)
            
            if 'azim_angle' not in self.data.columns:
                self.process_data(verbose, level+1, log_delim)
            
            # Shift
            angles = self.data.azim_angle - shift_sector
            angles[angles < 0] += 360

            angles = angles.sort_values()
            domain_id = np.ones((len(angles)))*-1
            i_pebble = 0
            are_ghosts = self.is_ghost()
            for i_domain in range(n_domains):
                cnt_domain = 0 
                while cnt_domain < target_npebbles_domains:
                    if not are_ghosts[angles.index[i_pebble]]:
                        domain_id[angles.index[i_pebble]] = i_domain
                        cnt_domain += 1
                    else:
                        domain_id[angles.index[i_pebble]] = np.nan
                    i_pebble += 1
                    
            # Handles last domain
            domain_id = np.array(domain_id)
            negative_id = domain_id < 0
            cnt_extra = np.sum(negative_id)
            domain_id[negative_id] = filling_domain
            if cnt_extra > 0:
                log_print(f'{cnt_extra} extra pebbles in domain {filling_domain}', verbose, level+1, log_delim)

        elif decomposition_type in ['4', 'radial', 'rad', 'r']:
            
            log_print(f'Decomposing pebbles by radial zones in {n_domains} domains', verbose, level, log_delim)

            if 'r_dist' not in self.data.columns:
                self.process_data(verbose, level+1, log_delim)
        
            radial_dist = self.data.r_dist.sort_values()
            domain_id = np.ones((len(radial_dist)))*-1
            i_pebble = 0
            are_ghosts = self.is_ghost()
            for i_domain in range(n_domains):
                cnt_domain = 0 
                while cnt_domain < target_npebbles_domains:
                    if not are_ghosts[radial_dist.index[i_pebble]]:
                        domain_id[radial_dist.index[i_pebble]] = i_domain
                        cnt_domain += 1
                    else:
                        domain_id[radial_dist.index[i_pebble]] = np.nan
                    i_pebble += 1

            # Handles last domain
            domain_id = np.array(domain_id)
            negative_id = domain_id < 0
            cnt_extra = np.sum(negative_id)
            domain_id[negative_id] = filling_domain
            if cnt_extra > 0:
                log_print(f'{cnt_extra} extra pebbles in domain {filling_domain}', verbose, level+1, log_delim)
        
        elif decomposition_type in ['5', 'axial', 'ax', 'a']:
            
            log_print(f'Decomposing pebbles by axial zones in {n_domains} domains', verbose, level, log_delim)        

            axial_dist = self.data.z.sort_values()
            domain_id = np.ones((len(axial_dist)))*-1
            i_pebble = 0
            are_ghosts = self.is_ghost()
            for i_domain in range(n_domains):
                cnt_domain = 0 
                while cnt_domain < target_npebbles_domains:
                    if not are_ghosts[axial_dist.index[i_pebble]]:
                        domain_id[axial_dist.index[i_pebble]] = i_domain
                        cnt_domain += 1
                    else:
                        domain_id[axial_dist.index[i_pebble]] = np.nan
                    i_pebble += 1

            # Handles last domain
            domain_id = np.array(domain_id)
            negative_id = domain_id < 0
            cnt_extra = np.sum(negative_id)
            domain_id[negative_id] = filling_domain
            if cnt_extra > 0:
                log_print(f'{cnt_extra} extra pebbles in domain {filling_domain}', verbose, level+1, log_delim)
        
        elif decomposition_type in ['6', 'spherical', 'spheric', 'sphere', 'sph', 'o']:

            log_print(f'Decomposing pebbles by spherical zones in {n_domains} domains', verbose, level, log_delim)        

            if 'dist' not in self.data.columns:
                self.process_data(verbose, level+1, log_delim)
        

            dist = self.data.dist.sort_values()
            domain_id = np.ones((len(dist)))*-1
            i_pebble = 0
            are_ghosts = self.is_ghost()
            for i_domain in range(n_domains):
                cnt_domain = 0 
                while cnt_domain < target_npebbles_domains:
                    if not are_ghosts[dist.index[i_pebble]]:
                        domain_id[dist.index[i_pebble]] = i_domain
                        cnt_domain += 1
                    else:
                        domain_id[dist.index[i_pebble]] = np.nan
                    i_pebble += 1

            # Handles last domain
            domain_id = np.array(domain_id)
            negative_id = domain_id < 0
            cnt_extra = np.sum(negative_id)
            domain_id[negative_id] = filling_domain
            if cnt_extra > 0:
                log_print(f'{cnt_extra} extra pebbles in domain {filling_domain}', verbose, level+1, log_delim)
        
        self.data['domain_id'] = domain_id


        if idle:
            self.domains = dict()
            _, cnt_domains = np.unique(self.data.domain_id, return_counts=True)
            for i_domain in range(n_domains):
                self.domains[i_domain] = cnt_domains[i_domain]
            log_print(f'Count in domains: {cnt_domains}', verbose, level+1, log_delim)

        log_print(f'Done.', verbose, level, log_delim, end_block=True)

        


def log_print(text, printing=True, level=0, log_delim='  ', line_limit=None, end_block=False, end_block_log_delim='\n', returning=True):
    # Line breaks
    if not isinstance(line_limit, type(None)):
        current_word = ''
        current_line = level*log_delim
        s = ''

        # Loop over message
        for c in text:
            if c != ' ':  # If not end of word
                if c == '\'':
                    c = '\"'
                current_word += c # Add to word
            else: # If end of word
                if len(current_line) + len(current_word) <= line_limit: # and if within line limits even when adding word
                    if current_line == level*log_delim:
                        current_line += current_word
                    else:
                        current_line += ' ' + current_word
                else: # if when adding word out of limits
                    s += current_line + '\n' # write line go to next line
                    current_line = level*log_delim + current_word # add word to next line
                current_word = '' # reset word
        s += current_line + ' ' + current_word # add last line
    else:
        s = level*log_delim + text

    if end_block and level==0:
        s += end_block_log_delim
    if printing:      
        print(s)
    if returning: 
        return s

#  if __name__ == '__main__':
    # verbose=True
    # folder = "/home/yryves/serpent_cases/test_larger/"
    # pbed_path = folder + "fpb_pos"
    # det0_path = folder + "input_det0.m"
    # dep_path  = folder + "input_dep.m"   
    # plotxy_path  = folder + "input_geom3.png"
    # plotyz_path  = folder + "input_geom4.png"
    # R = 80
    # H = 190
    # dd=True
    # fuel_material='fuel' 
    # pbed = Pebble_bed()
    # pbed.read_file(pbed_path, verbose=verbose)   
    # dz = 10
    # ori_r = np.array(pbed.data.r)
    # for i in range(10):
        # pbed.data.z += dz
        # pbed.data.loc[pbed.data.z + ori_r > 150, "z"] -= (150-40)
        # pbed.data.loc[pbed.data.z - ori_r < 40, "r"] = 0
        # pbed.data.loc[pbed.data.z - ori_r >= 40, "r"] = ori_r[pbed.data.z - ori_r >= 40] 
    # plt.figure()
    # ax= pbed.show_Serpent_plot(plotxy_path, xlim=[-R,R], ylim=[-R,R], verbose=verbose) #, tol=1000)
    # plt.show()
# 
    # plt.figure()
    # ax= pbed.show_Serpent_plot(plotyz_path, xlim=[-R,R], ylim=[0,H], verbose=verbose) #, tol=1000)
    # plt.show()
    # ax= pbed.plot2D(field='z', verbose=verbose) #, tol=1000)
    # plt.show()   
    # pbed.read_detector(det0_path, which_dets='all', verbose=verbose)
    # ax= pbed.plot2D(field='id', verbose=verbose) #, tol=1000)
    # plt.show()   
    # pbed.read_depletion(dep_path, material_name=fuel_material, dd=dd, fields='all', verbose=verbose)
    # ax = pbed.plot3D(force_slow=True, sample_fraction=1, field='id', verbose=verbose)
    # plt.show()
    # error    
    # pbed.clip('rdist', 150, +1).clip('rdist', 155, -1) 
    # plt.figure()
    # ax = pbed.plot2D(field='dist', field_title='Distance to center')
    # plt.show() 
    # plt.figure()
    # ax= pbed.plot2D(field='r_dist', field_title='Radial distance to center')
    # plt.show() 

verbose=True
folder = "/home/yryves/serpent_cases/test_larger/"
pbed_path = folder + "fpb_pos"
pbed = Pebble_bed()
pbed.read_file(pbed_path, verbose=verbose)  

for t in ['as', 'ar', 'si', 'nn', 'ii', 'ao']:
    pbed.decompose_in_domains([2,4], t, verbose=False)
    pbed.plot3D('domain_id', fig_size=(6,8), scatter_size=50, save_fig=True, fig_suffix=f'_{t}', field_title=f'DD: {t}', verbose=False)
    plt.show()

for t in ['asr', 'aro', 'sir', 'nao', 'rao']:
    pbed.decompose_in_domains([2,2,2], t, verbose=False)
    pbed.plot3D('domain_id', fig_size=(6,8), scatter_size=50, save_fig=True, fig_suffix=f'_{t}', field_title=f'DD: {t}', verbose=False)
    plt.show()

for t in ['asro']:
    pbed.decompose_in_domains([2,2,2,2], t, verbose=False)
    pbed.plot3D('domain_id', fig_size=(6,8), scatter_size=50, save_fig=True, fig_suffix=f'_{t}', field_title=f'DD: {t}', verbose=False)
    plt.show()
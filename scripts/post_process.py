import os
from keras.models import load_model
from sealsml.data import Preprocessor
from bridgescaler import load_scaler
from sealsml.geometry import polar_to_cartesian
import xarray as xr
import pandas as pd
import numpy as np
from os.path import join

### MODEL BASE PATH, OUT PATH, AND TIME STAMP
base_path = "/glade/derecho/scratch/jsauer/SEALS_TRAINING/TEST_TRANSFORMER/"
run_time = "2024-09-13_1959"
base_out_path = "/glade/derecho/scratch/cbecker/seals_evaluation/"
# base_path ="/glade/derecho/scratch/cbecker/SEALS_output/"
# run_time = "2024-11-07_1432"
model_type = "loc_rate_block_transformer"
out_path = os.path.join(base_path, run_time)
os.makedirs(out_path, exist_ok=True)
coord_data_path = "/glade/derecho/scratch/jsauer/ForPeople/ForCAMS/Phase_II/OrigSets/FE_CBL_00_RefOri.nc"

## VARS USED TO REVERSE ENGINEER MAPPING OF POTENTIAL LEAK LOC TO EQUIPMENT PIECE
rot_names = ['RotCW0','RotCW60','RotCW180','RotCW225','RotCW285']
rot_angles = [0,60,180,225,285]
rot_path = '/glade/derecho/scratch/jsauer/ForPeople/ForCAMS/FE_METEC_potleaks/'
rot_root = 'METEC_EquipLevelPotLeaks_v2_'

## EQUIPMENT MAPPINGS
group = {"3W": [0, 1, 2], "4W": [3, 4, 5, 6, 7], "5S": [9, 10, 11], "4T": [12, 13,14], "3S": [15, 16, 17],
          "4S": [18, 19, 20, 21], "5W": [23, 24, 25], "3T": [26, 27], "2TWS": [29, 30, 31],
           "1WT": [34, 35, 36], "Misc": [8, 22, 28, 32, 33], "No Match": [37, 999]}
equip_type = {"W": [0, 1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 30, 35], "S": [9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 31, 34],
              "T": [12, 13, 14, 26, 27, 29, 36], "Misc": [8, 22, 28, 32, 33], "No Match": [37, 999]}

ds_coords=xr.open_dataset(coord_data_path)
xc = ds_coords['xPos'][0, 0, :].values
yc = ds_coords['yPos'][0, :, 0].values
zc = ds_coords['zPos'][:, 0, 0].values

p = Preprocessor(sensor_type_value=-999, sensor_pad_value=-1)
val_files = pd.read_csv(join(base_path, run_time, 'validation_files.csv'))
encoder_data_val, decoder_data_val, leak_location_val, leak_rate_val = p.load_data(val_files['validation_files'],
                                                                                   remove_blind_samples=False,
                                                                                   use_noise=False)

p.sensor_scaler = load_scaler(join(base_path, run_time, "sensor_scaler.json"))
p.coord_scaler = load_scaler(join(base_path, run_time, "coord_scaler.json"))

scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val = p.preprocess(encoder_data_val,
                                                                                          decoder_data_val,
                                                                                          fit_scaler=False)
model = load_model(join(base_path, run_time, f"{model_type}_{run_time}.keras"))

preds_val, preds_val_lr = model.predict((scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val), batch_size=2056)

def windRelPolarToOrigCartesian(targetDist, targetSinAz, targetCosAz, targetElAng, mean_wd, x0, y0, z0):
    # x0 -- original cartesian x-coordinate of the reference (met) sensor
    # y0 -- original cartesian y-coordinate of the reference (met) sensor
    # z0 -- original cartesian z-coordinate of the reference (met) sensor
    x, y = polar_to_cartesian(targetDist, targetSinAz, targetCosAz)
    x_orig = x * np.cos(mean_wd) - y * np.sin(mean_wd) + x0
    y_orig = x * np.sin(mean_wd) + y * np.cos(mean_wd) + y0
    z_orig = targetDist * np.sin(targetElAng) + z0

    return x_orig, y_orig, z_orig


def findIndices(xloc, yloc, zloc, xc, yc, zc):
    i_indx = np.abs(xc - xloc).argmin()
    j_indx = np.abs(yc - yloc).argmin()
    k_indx = np.abs(zc - zloc).argmin()

    return i_indx, j_indx, k_indx


def findExtraPL(rot_coords, xcoord, ycoord, zcoord):
    verbose = False
    dec_prec = 4
    xextra_vals = np.setdiff1d(np.round(xcoord, decimals=dec_prec), np.round(rot_coords[:, 0],
                                                                             decimals=dec_prec))  # Order matters here to find the element in 1st array that is not in the second (won't work reversed)
    yextra_vals = np.setdiff1d(np.round(ycoord, decimals=dec_prec), np.round(rot_coords[:, 1],
                                                                             decimals=dec_prec))  # Order matters here to find the element in 1st array that is not in the second (won't work reversed)
    zextra_vals = np.setdiff1d(np.round(zcoord, decimals=dec_prec), np.round(rot_coords[:, 2],
                                                                             decimals=dec_prec))  # Order matters here to find the element in 1st array that is not in the second (won't work reversed)

    if verbose:
        print(xextra_vals)
        print(yextra_vals)
        print(zextra_vals)

    if xextra_vals.size > 0:
        xextra_indx = np.argwhere(np.round(xcoord, decimals=dec_prec) == xextra_vals[0])
        if yextra_vals.size > 0:
            yextra_indx = np.argwhere(np.round(ycoord, decimals=dec_prec) == yextra_vals[0])
        if zextra_vals.size > 0:
            zextra_indx = np.argwhere(np.round(zcoord, decimals=dec_prec) == zextra_vals[0])

        if xextra_indx.size > 0:
            ret_indx = xextra_indx[0][0]
        else:
            if yextra_indx.size > 0:
                ret_indx = yextra_indx[0][0]
            elif zextra_indx.size > 0:
                ret_indx = zextra_indx[0][0]


    elif yextra_vals.size > 0:
        yextra_indx = np.argwhere(np.round(ycoord, decimals=dec_prec) == yextra_vals[0])
        if verbose:
            print(f"{yextra_vals[0]}: {np.argwhere(np.round(ycoord, decimals=dec_prec) == yextra_vals[0])}")
        ret_indx = yextra_indx[0][0]
    elif zextra_vals.size > 0:
        zextra_indx = np.argwhere(np.round(zcoord, decimals=dec_prec) == zextra_vals[0])
        ret_indx = zextra_indx[0][0]
    else:
        print(f"No extra val...")
        print(np.sort(np.round(rot_coords[:, 0], decimals=dec_prec)))
        print(np.sort(np.round(xcoord, decimals=dec_prec)))
        ret_indx = 37
    return ret_indx


def findPlIndxMapping(rot_coords, xcoord, ycoord):
    if False:
        rxc_inds = np.argsort(rot_coords[:, 0])
        xc_inds = np.argsort(xcoord)
    else:
        rxc_inds = np.argsort(np.sqrt(rot_coords[:, 0] ** 2 + rot_coords[:, 1] ** 2))
        xc_inds = np.argsort(np.sqrt(xcoord ** 2 + ycoord ** 2))
    sorted_inds = np.argsort(rxc_inds)
    return xc_inds, xc_inds[sorted_inds], sorted_inds

def equip_map(dict, equip_index):
    for name, group in dict.items():
        if equip_index in group:
            return name



equip_group_table = np.zeros([encoder_data_val.shape[0],decoder_data_val.shape[1]],dtype=np.int32)
equip_group_table_unsorted = np.zeros([encoder_data_val.shape[0],decoder_data_val.shape[1]],dtype=np.int32)
equip_group_map = np.zeros([encoder_data_val.shape[0],decoder_data_val.shape[1]],dtype=np.int32)


# Read in the sensor_loc (x,y,z) from the potential leak files (6: original + 4 rotations)
plDim=0
ds_rr=[]
for rr in range(0,len(rot_names)):
    file_rr = f"{rot_path}{rot_root}{rot_names[rr]}.nc"
    print(file_rr)
    ds_rr.append(xr.open_dataset(file_rr))
    plDim=max(plDim,ds_rr[rr].sizes['plDim'])
locDim=ds_rr[rr].sizes['locDim']
rot_coords_sets = np.zeros(shape=(len(rot_names),plDim,locDim))
for rr in range(0,len(rot_names)):
    rot_coords_sets[rr,:,:] = ds_rr[rr].srcPotLeakLocation.values

met_id = 0
dec_prec = 4
verbose = False

xpl = 0
xpl_i = []
xpl_ix = []
extra_pl_flag = []
delta_nans = []
## For each of the validation files
ff_tot = 0  # 7938 #1782 ## Can be used to check particular files and samples sets

for ff in range(0, len(val_files)):

    file_ff = val_files.iloc[ff, 1]

    if verbose:
        print(f"{ff}: {file_ff}")
    extraPL = '_set' not in file_ff
    if verbose:
        print(extraPL)
    case_ff_m1 = file_ff.split('_')[-1]
    if verbose:
        print(f"case_ff_m1 = {case_ff_m1}")

    if (case_ff_m1 == 'Sensit.nc'):
        str_sect = -3
    else:
        str_sect = -2
    case_ff = file_ff.split('_')[str_sect]
    if verbose:
        print(f"case_ff = {case_ff}")

    if 'RefOri' in case_ff:
        rotAngle = 0
    else:
        rotAngle = np.int32(case_ff.split('RotCW')[-1])
    angleIndx = rot_angles.index(rotAngle)
    if verbose:
        print(f"{rotAngle} --> {angleIndx}")
    for ii in range(0,len(rot_names)):
       rot_ii = rot_names[ii]
       if (case_ff==rot_ii):
           ind_rot = ii
           break


    ds_ff = xr.open_dataset(file_ff)

    # Number of samples in this file
    sample_ff = ds_ff.sizes['sample']
    # Original Cartesian coordinate mean wind direction in radians [per-sample]
    mean_wd_ff = ds_ff['mean_wd'].values
    # Original Cartesian coordinates of the sensors [per-sample]
    sensor_coords_ff = ds_ff['sensor_meta'].values
    ff_tot_s = ff_tot
    ff_tot_e = ff_tot + sample_ff
    if verbose:
        print(f"{ds_ff.sizes['sample']} samples, ff_tot_s = {ff_tot_s} - {ff_tot_e - 1}")

    # reference pot leak coordinates

    if extraPL:
        rot_coords_xy_tmp = np.zeros(shape=(plDim, locDim))
        rot_coords_xy_ff = np.zeros(shape=(plDim + 1, locDim))

    else:
        rot_coords_xy_ff = np.zeros(shape=(plDim, locDim))

    for ss in range(0, sample_ff):  # loop across samples in each file
        # Establish the Original carteisan coordinates of the grid-"fit" (i.e. set by index) pl cartesian coorinates from the decoder
        xcoord_ss, ycoord_ss, zcoord_ss = windRelPolarToOrigCartesian(decoder_data_val[ff_tot_s + ss, :, 0, 0, 0].values,
                                                                      decoder_data_val[ff_tot_s + ss, :, 0, 1, 0].values,
                                                                      decoder_data_val[ff_tot_s + ss, :, 0, 2, 0].values,
                                                                      decoder_data_val[ff_tot_s + ss, :, 0, 3, 0].values,
                                                                      ds_ff['mean_wd'][ss].values,
                                                                      ds_ff['met_sensor_loc'][ss, 0].values,
                                                                      ds_ff['met_sensor_loc'][ss, 1].values,
                                                                      ds_ff['met_sensor_loc'][ss, 2].values)
        if ss == 0:  # On the first sample for this file
            # Establish a set of grid-"fit" (i.e. set by index) pl cartesian coorinates from the original set of specified-by-file pls
            for ipl in range(rot_coords_sets.shape[1]):
                i_ind, j_ind, k_ind = findIndices(rot_coords_sets[angleIndx, ipl, 0],
                                                  rot_coords_sets[angleIndx, ipl, 1],
                                                  rot_coords_sets[angleIndx, ipl, 2],
                                                  xc, yc, zc)
                rot_coords_xy_tmp[ipl, 0] = ds_coords['xPos'][0, 0, i_ind].values
                rot_coords_xy_tmp[ipl, 1] = ds_coords['yPos'][0, j_ind, 0].values
                rot_coords_xy_tmp[ipl, 2] = ds_coords['zPos'][k_ind, 0, 0].values
                rot_coords_xy_ff[ipl, 0] = ds_coords['xPos'][0, 0, i_ind].values
                rot_coords_xy_ff[ipl, 1] = ds_coords['yPos'][0, j_ind, 0].values
                rot_coords_xy_ff[ipl, 2] = ds_coords['zPos'][k_ind, 0, 0].values
            if not extraPL:  # There is no need to account for any extra pl in every sample
                rot_coords_xy_ff = rot_coords_xy_tmp
        # If there is an "extra pl" corresponding to a source that was not in specified in the pls from file
        # Add the unique coordinates of it to the 37th element of the non-randomized list of pls from file
        if extraPL:
            rot_coords_xy_ff[:-1, :] = rot_coords_xy_tmp
            extraPL_indx = findExtraPL(rot_coords_xy_tmp, xcoord_ss, ycoord_ss, zcoord_ss)
            # Set the rot_coord 37th pl-dim element to the extraPL coordinates
            rot_coords_xy_ff[-1, 0] = np.round(xcoord_ss[extraPL_indx], decimals=dec_prec)
            rot_coords_xy_ff[-1, 1] = np.round(ycoord_ss[extraPL_indx], decimals=dec_prec)
            rot_coords_xy_ff[-1, 2] = np.round(zcoord_ss[extraPL_indx], decimals=dec_prec)
            xpl_i.append(xpl)
            xpl_ix.append(extraPL_indx)
            extra_pl_flag.append(1)
        else:
            extra_pl_flag.append(0)
        xpl_inds, xpl_inds_sorted, sorted_inds = findPlIndxMapping(rot_coords_xy_ff, xcoord_ss, ycoord_ss)

        if xpl_inds_sorted.shape[0] == equip_group_table.shape[1]:  # same as if extra_PL: ??
            equip_group_table[xpl, :] = xpl_inds_sorted
            equip_group_table_unsorted[xpl, :] = xpl_inds
            equip_group_map[xpl, :] = xpl_inds_sorted

            # Map any extra PLs to closest eq/grp pl-index from pl-spec file as appropriate
            dpl_coords = np.asarray(
                [xcoord_ss[xpl_inds_sorted[37]], ycoord_ss[xpl_inds_sorted[37]], zcoord_ss[xpl_inds_sorted[37]]])
            if verbose:
                print(dpl_coords)
            delta_vec = np.sqrt(np.sum((rot_coords_xy_ff[:-1, :] - dpl_coords.T) ** 2, axis=1))
            plindx_nearest = np.argsort(delta_vec)[0]
            if verbose:
                print(delta_vec)
                print(plindx_nearest)
            delta_nans.append(delta_vec)
            equip_group_map[xpl, 37] = xpl_inds_sorted[plindx_nearest]
        else:
            equip_group_table[xpl, :-1] = xpl_inds_sorted
            equip_group_table[xpl, -1] = 37
            equip_group_table_unsorted[xpl, :] = xpl_inds
            equip_group_map[xpl, :-1] = xpl_inds_sorted
            equip_group_map[xpl, -1] = 37

        xpl += 1

    ff_tot = ff_tot + sample_ff


# x / equip_num_true is unique equipment ID for the TRUE Leak location
equip_num_true = []
adjusted_true=[]
true_leak_i = np.argmax(leak_location_val, axis=1)

for i in range(equip_group_map.shape[0]):
    new_true = equip_group_map[i, np.argwhere(equip_group_table[i] == true_leak_i[i]).min()]
    true_ID = np.argwhere(equip_group_map[i] == new_true)
    equip_num_true.append(true_ID.min())
    adjusted_true.append(new_true)

# x / equip_num_pred is unique equipment ID for the PRED Leak location
equip_num_pred = []
adjusted_pred = []
pred_leak_i = np.argmax(preds_val, axis=1)
for i in range(equip_group_map.shape[0]):
    new_pred = equip_group_map[i, np.argwhere(equip_group_table[i] == pred_leak_i[i]).min()]
    pred_ID = np.argwhere(equip_group_map[i] == new_pred)
    equip_num_pred.append(pred_ID.min())
    adjusted_pred.append(new_pred)

target_loc = leak_location_val.argmax(axis=1)
pred_loc = preds_val.argmax(axis=1)
t_ID = np.array(equip_num_true)
p_ID = np.array(equip_num_pred)


n_samples = equip_group_table.shape[0]
true_equip_type = list(map(equip_map, [equip_type] * n_samples, equip_num_true))
true_group = list(map(equip_map, [group] * n_samples, equip_num_true))
pred_equip_type = list(map(equip_map, [equip_type] * n_samples, equip_num_pred))
pred_group = list(map(equip_map, [group] * n_samples, equip_num_pred))

ds = xr.merge([encoder_data_val,
              decoder_data_val.squeeze(),
              xr.DataArray(leak_location_val, dims=('sample', 'pot_leak'), name='target_leak_loc'),
              xr.DataArray(leak_rate_val, dims=('sample'), name='target_leak_rate'),
              xr.DataArray(preds_val, dims=('sample','pot_leak'), name='leak_loc_preds'),
              xr.DataArray(preds_val_lr.squeeze(), dims=('sample'), name='leak_rate_pred'),
              xr.DataArray(equip_group_table, dims=('sample', 'pot_leak'), name='equipment_table'),
              xr.DataArray(equip_group_map, dims=('sample','pot_leak'), name='equipment_map'),
              xr.DataArray(t_ID, dims=('sample'), name='true_equip_ID'),
              xr.DataArray(p_ID, dims=('sample'), name='pred_equip_ID'),
              xr.DataArray(np.array(true_group), dims=('sample'), name='true_group'),
              xr.DataArray(true_equip_type, dims=('sample'), name='true_equip_type'),
              xr.DataArray(pred_group, dims=('sample'), name='pred_group'),
              xr.DataArray(pred_equip_type, dims=('sample'), name='pred_equip_type'),
              xr.DataArray(np.array(extra_pl_flag), dims=('sample'), name='extra_PL_flag')])

out_file_path = join(out_path, f"model_output_{run_time}.nc")
ds.to_netcdf(out_file_path)
print(f"Completed writing {out_file_path}")



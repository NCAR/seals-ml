import xarray as xr
from keras.models import load_model
from sealsml.data import Preprocessor
import numpy as np
import pandas as pd
from bridgescaler import load_scaler
from os.path import join
from sealsml.geometry import polar_to_cartesian

def windRelPolarToOrigCartesian(targetDist, targetSinAz, targetCosAz, targetElAng, mean_wd, x0, y0, z0):
    # x0 -- original cartesian x-coordinate of the reference (met) sensor
    # y0 -- original cartesian y-coordinate of the reference (met) sensor
    # z0 -- original cartesian z-coordinate of the reference (met) sensor
    x,y=polar_to_cartesian(targetDist,targetSinAz,targetCosAz)
    x_orig = x*np.cos(mean_wd) - y*np.sin(mean_wd) + x0
    y_orig = x*np.sin(mean_wd) + y*np.cos(mean_wd) + y0
    z_orig = targetDist*np.sin(targetElAng) + z0

    return x_orig, y_orig, z_orig

def equip_map(dict, equip_index):
    for name, group in dict.items():
        if equip_index in group:
            return name

### MODEL BASE PATH, OUT PATH, AND TIME STAMP
base_path = "/glade/derecho/scratch/jsauer/SEALS_TRAINING/TEST_TRANSFORMER/"
run_time = "2024-08-22_0927"
out_path = f"/glade/derecho/scratch/cbecker/model_eval_output_{run_time}.nc"

## VARS USED TO REVERSE ENGINEER MAPPING OF POTENTIAL LEAK LOC TO EQUIPMENT PIECE
rot_names = ['RefOri','RotCW0','RotCW180','RotCW225','RotCW285','RotCW60']
rot_path = '/glade/derecho/scratch/jsauer/ForPeople/ForCAMS/FE_METEC_potleaks/'
rot_root = 'METEC_EquipLevelPotLeaks_v2_'

## EQUIPMENT MAPPINGS
group = {"3W": [0, 1, 2], "4W": [3, 4, 5, 6, 7], "8": [8], "5S": [9, 10, 11], "4T": [12, 13,14], "3S": [15, 16, 17],
          "4S": [18, 19, 20, 21], "22": [22], "5W": [23, 24, 25], "3T": [26, 27], "28": [28], "2TWS": [29, 30, 31],
           "32": [32], "33": [33], "1WT": [34, 35, 36]}
equip_type = {"W": [0, 1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 30, 35], "S": [9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 31, 34],
              "T": [12, 13, 14, 26, 27, 29, 36], "Misc": [8, 22, 28, 32, 33]}


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
model = load_model(join(base_path, run_time, f"loc_rate_block_transformer_{run_time}.keras"))

preds_val = model.predict((scaled_encoder_val, scaled_decoder_val, encoder_mask_val, decoder_mask_val), batch_size=256)

equip_group_table = np.zeros([encoder_data_val.shape[0], decoder_data_val.shape[1]],dtype=np.int32)
nf_val = len(val_files)

rot_coords_sets = np.zeros([len(rot_names),decoder_data_val.shape[1]-1,3]) # sticking to the 37 potential leak locations...

for rr in range(0,len(rot_names)):
    file_rr = rot_path + rot_root + rot_names[rr] + '.nc'
    ds_rr = xr.open_dataset(file_rr)
    rot_coords_sets[rr,:,:] = ds_rr.srcPotLeakLocation.values

met_id = 0

ff_tot = 0
for ff in range(0, nf_val):

    file_ff = val_files.iloc[ff, 1]

    case_ff_m1 = file_ff.split('_')[-1]

    if (case_ff_m1 == 'Sensit.nc'):
        str_sect = -3
    else:
        str_sect = -2
    case_ff = file_ff.split('_')[str_sect]

    for ii in range(0, len(rot_names)):
        rot_ii = rot_names[ii]
        if (case_ff == rot_ii):
            ind_rot = ii
            break

    ds_ff = xr.open_dataset(file_ff)

    sample_ff = ds_ff.sample.size
    mean_wd_ff = ds_ff.mean_wd.values
    sensor_coords_ff = ds_ff.sensor_meta.values

    ff_tot_s = ff_tot
    ff_tot_e = ff_tot + sample_ff

    # reference pot leak coordinates
    rot_coords_xy_ff = rot_coords_sets[ind_rot, :, 0:2]
    for ss in range(0, sample_ff):  # loop across samples in each file

        xcoord_ss, ycoord_ss, zcoord_ss = windRelPolarToOrigCartesian(
            decoder_data_val[ff_tot_s + ss, :, 0, 0, 0].values,
            decoder_data_val[ff_tot_s + ss, :, 0, 1, 0].values,
            decoder_data_val[ff_tot_s + ss, :, 0, 2, 0].values,
            decoder_data_val[ff_tot_s + ss, :, 0, 3, 0].values,
            mean_wd_ff[ss],
            sensor_coords_ff[ss, met_id, 0],
            sensor_coords_ff[ss, met_id, 1],
            sensor_coords_ff[ss, met_id, 2])

        ind_pp_v = np.zeros(rot_coords_sets.shape[1] + 1, dtype=np.int32)
        min_diff_v = np.zeros(rot_coords_sets.shape[1] + 1)
        for pp in range(0, rot_coords_sets.shape[1]):

            xy_diff_pp = np.sqrt(
                np.power(xcoord_ss[pp] - rot_coords_xy_ff[:, 0], 2.0) + np.power(ycoord_ss[pp] - rot_coords_xy_ff[:, 1],
                                                                                 2.0))

            if (np.isnan(np.nanmin(xy_diff_pp)) == False):

                ind_min_pp = np.where(np.nanmin(xy_diff_pp) == xy_diff_pp)
                ind_pp = ind_min_pp[0][0]
                ind_pp_v[pp] = ind_pp
                min_diff_v[pp] = np.nanmin(xy_diff_pp)

            else:
                ind_pp_v[pp] = 37
                min_diff_v[pp] = 0.5

            if (min_diff_v[pp] > 0.5):
                ind_pp_v[pp] = 37
                min_diff_v[pp] = 0.49


        equip_group_table[ff_tot_s + ss, :] = ind_pp_v

    ff_tot = ff_tot + sample_ff

n_samples = equip_group_table.shape[0]
equip_types = list(map(equip_map, [equip_type] * n_samples,
                       equip_group_table[np.arange(np.argmax(leak_location_val, axis=1).size),
                       np.argmax(leak_location_val, axis=1)]))
groups = list(map(equip_map, [group]*151848,
                  equip_group_table[np.arange(np.argmax(leak_location_val, axis=1).size),
                  np.argmax(leak_location_val, axis=1)]))

merged_ds = xr.merge([encoder_data_val,
                      decoder_data_val.squeeze(),
                      xr.DataArray(leak_location_val, dims=('sample', 'pot_leak'), name='target_leak_loc'),
                      xr.DataArray(leak_rate_val, dims=('sample'), name='target_leak_rate'),
                      xr.DataArray(preds_val[0], dims=('sample','pot_leak'), name='leak_loc_preds'),
                      xr.DataArray(preds_val[1].squeeze(), dims=('sample'), name='leak_rate_pred'),
                      xr.DataArray(equip_group_table, dims=('sample','pot_leak'), name='equipment_num'),
                      xr.DataArray(groups, dims=('sample'), name='target_group'),
                      xr.DataArray(equip_types, dims=('sample'), name='target_equip_type'),
                     ])

merged_ds.to_netcdf(out_path)
print(f"Completed writing {out_path}")
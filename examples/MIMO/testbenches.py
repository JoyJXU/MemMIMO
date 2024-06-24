from utility import utility
import time
import os
import torch
import pickle
from simbrain.memarray import MemristorArray


def run_d2d_sim(_crossbar_1, _crossbar_2, _crossbar_3, _crossbar_4, _rep, _batch_size, _rows, _cols, sim_params, device, _logs=[None, None, False, False, None]):
    print("<========================================>")
    print("Test case: ", _rep)
    file_name = "d2d_test_case_r" + str(_rows) + "_c" + str(_cols) + "_rep" + str(_rep) + ".csv"
    file_path = _logs[0]  # main file path
    header = ['size', 'G_sigma', 'r']
    file = file_path + "/" + file_name  # Location to the file for the main results
    # Only write header once
    if not (os.path.isfile(file)):
        utility.write_to_csv(file_path, file_name, header)

    print("<==============>")
    start_time = time.time()
    print("Row No. ", _rows, " Column No. ", _cols, " Repetition No. ", _rep, " Batch Size: ", _batch_size)

    print("<==============>")
    sigma_list_g = [0,0.0005,0.001,0.005,0.01,0.05,0.1, 0.5, 1]
    sigma_list_p = [0.0005]
    # print("Start Sigma: ", sigma_list[1], ", End Sigma: ", sigma_list[-1], ", Sigma=0 Included")

    _var_g = 0
    _var_linearity = 0
    no_trial = 5
    read_batch = 1
    for _var_g in sigma_list_g:
        for _var_linearity in sigma_list_p:
            trial_error = torch.zeros(5,device=device)
            for trial in range(no_trial):
                device_name = sim_params['device_name']
                input_bit = sim_params['input_bit']
                batch_interval = 1 + _crossbar_1.memristor_luts[device_name][
                    'total_no'] * _rows + 1 * input_bit  # reset + write + read
                batch_interval = 1 + _crossbar_2.memristor_luts[device_name][
                    'total_no'] * _rows + 1 * input_bit  # reset + write + read
                batch_interval = 1 + _crossbar_3.memristor_luts[device_name][
                    'total_no'] * _rows + 1 * input_bit  # reset + write + read
                batch_interval = 1 + _crossbar_4.memristor_luts[device_name][
                    'total_no'] * _rows + 1 * input_bit  # reset + write + read
                _crossbar_1.batch_interval = batch_interval
                _crossbar_2.batch_interval = batch_interval
                _crossbar_3.batch_interval = batch_interval                
                _crossbar_4.batch_interval = batch_interval
                
                # Perform d2d variation only
                sim_params['c2c_variation'] = False
                sim_params['d2d_variation'] = 1
                memristor_info_dict = _crossbar_1.memristor_info_dict
                memristor_info_dict[device_name]['Gon_sigma'] = _var_g
                memristor_info_dict[device_name]['Goff_sigma'] = _var_g

                memristor_info_dict[device_name]['Pon_sigma'] = _var_linearity
                memristor_info_dict[device_name]['Poff_sigma'] = _var_linearity

                _crossbar_1.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_1.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_1.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_1.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                                         memristor_info_dict=memristor_info_dict)
        
                _crossbar_2.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_2.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_2.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_2.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                                         memristor_info_dict=memristor_info_dict)
        
                _crossbar_3.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_3.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_3.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_3.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                                         memristor_info_dict=memristor_info_dict)
        
                _crossbar_4.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_4.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_4.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_4.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_1.to(device)
                _crossbar_1.set_batch_size_mimo(_batch_size)
        
                _crossbar_2.to(device)
                _crossbar_2.set_batch_size_mimo(_batch_size)
        
                _crossbar_3.to(device)
                _crossbar_3.set_batch_size_mimo(_batch_size)
        
                _crossbar_4.to(device)
                _crossbar_4.set_batch_size_mimo(_batch_size)
        
                # matrix and vector random generation
                # matrix = torch.rand(_rep, _rows, _cols, device=device)
                matrix_r = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
                matrix_i = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
                # matrix = torch.ones(_rep, _rows, _cols, device=device)
                # vector = torch.rand(_rep, 1, _rows, device=device)
                vector_r = -1 + 2 * torch.rand(_rep, read_batch, _rows, device=device)
                vector_i = -1 + 2 * torch.rand(_rep, read_batch, _rows, device=device)
                # vector = torch.ones(_rep, 1, _rows, device=device)
                # print("Randomized input")
        
                # Golden results calculation
                golden_model_1 = torch.matmul(vector_r, matrix_r)
                golden_model_2 = torch.matmul(vector_r, matrix_i)
                golden_model_3 = torch.matmul(vector_i, matrix_r)
                golden_model_4 = torch.matmul(vector_i, matrix_i)
                golden_r = golden_model_1 - golden_model_4
                golden_i = golden_model_2 + golden_model_3
    
                n_step = int(_rep / _batch_size)
                cross_1 = torch.zeros_like(golden_model_1, device=device)
                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix_r[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector_r[(step * _batch_size):(step * _batch_size + _batch_size)]
        
                    # Memristor-based results simulation
                    # Memristor crossbar program
                    _crossbar_1.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross_1[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_1.mapping_read_mimo(
                        target_v=vector_batch)
        
                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar_1.total_energy_calculation()
                        sim_power = _crossbar_1.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)
        
                    # mem_t update # Avoid mem_t at the last batch
                    if not step == n_step - 1:
                        _crossbar_1.mem_t_update()
        
                cross_2 = torch.zeros_like(golden_model_2, device=device)
                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix_i[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector_r[(step * _batch_size):(step * _batch_size + _batch_size)]
        
                    # Memristor-based results simulation
                    # Memristor crossbar program
                    _crossbar_2.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross_2[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_2.mapping_read_mimo(
                        target_v=vector_batch)
        
                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar_2.total_energy_calculation()
                        sim_power = _crossbar_2.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)
        
                    # mem_t update # Avoid mem_t at the last batch
                    if not step == n_step - 1:
                        _crossbar_2.mem_t_update()
        
                cross_3 = torch.zeros_like(golden_model_3, device=device)
                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix_r[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector_i[(step * _batch_size):(step * _batch_size + _batch_size)]
        
                    # Memristor-based results simulation
                    # Memristor crossbar program
                    _crossbar_3.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross_3[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_3.mapping_read_mimo(
                        target_v=vector_batch)
        
                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar_3.total_energy_calculation()
                        sim_power = _crossbar_3.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)
        
                    # mem_t update # Avoid mem_t at the last batch
                    if not step == n_step - 1:
                        _crossbar_3.mem_t_update()
        
                cross_4 = torch.zeros_like(golden_model_4, device=device)
                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix_i[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector_i[(step * _batch_size):(step * _batch_size + _batch_size)]
        
                    # Memristor-based results simulation
                    # Memristor crossbar program
                    _crossbar_4.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross_4[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_4.mapping_read_mimo(
                        target_v=vector_batch)
        
                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar_4.total_energy_calculation()
                        sim_power = _crossbar_4.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)
        
                    # mem_t update # Avoid mem_t at the last batch
                    if not step == n_step - 1:
                        _crossbar_4.mem_t_update()
        
                delta_cross_r = cross_1 - cross_4 - golden_r
                delta_cross_i = cross_2 + cross_3 - golden_i
                
                cross_error = torch.sqrt(
                    torch.sum(torch.square(delta_cross_r)) + torch.sum(torch.square(delta_cross_i))) / torch.sqrt(
                    torch.sum(torch.square(golden_r)) + torch.sum(torch.square(golden_i)))
        
                torch.set_printoptions(precision=8)
                trial_error[trial] = cross_error
                print("Gon/Goff Sigma: ", _var_g, ", Nonlinearity Sigma: ", _var_linearity, ", Mean Error: ", cross_error)
            print("trial_mean_error:",torch.mean(trial_error))
    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)


def run_c2c_sim(_crossbar_1, _crossbar_2, _crossbar_3, _crossbar_4, _rep, _batch_size, _rows, _cols, sim_params, device,
                _logs=[None, None, False, False, None], figs=None):
    print("<========================================>")
    print("Test case: ", _rep)
    file_name = "crossbar_size_test_case_r" + str(_rows) + "_c" + \
                str(_cols) + "_rep" + str(_rep) + ".csv"
    file_path = _logs[0]  # main file path
    header = ['size', 'AB_sigma', 'RE_sigma', 'me', 'mae', 'rmse', 'rmae', 'rrmse1', 'rrmse2', 'rpd1', 'rpd2', 'rpd3',
              'rpd4']
    file = file_path + "/" + file_name  # Location to the file for the main results
    # Only write header once
    if not (os.path.isfile(file)):
        utility.write_to_csv(file_path, file_name, header)

    print("<==============>")
    start_time = time.time()

    # Batch Size Adaption
    if (_batch_size * _rows) > 2e6 and _batch_size >= 10:
        _batch_size = int(_batch_size / 10)
    elif (_batch_size * _rows) < 2e5 and _batch_size <= (_rep / 10):
        _batch_size = int(_batch_size * 10)

    print("Row No. ", _rows, " Column No. ", _cols, " Repetition No. ", _rep, " Batch Size: ", _batch_size)

    print("<==============>")
    sigma_list_re = [0.0005]
    sigma_list_ab = [0,0.0005,0.001,0.005,0.01, 0.05, 0.1, 0.5, 1]
    _var_abs = 0
    _var_rel = 0
    no_trial = 5
    read_batch = 1
    for _var_abs in sigma_list_ab:
        for _var_rel in sigma_list_re:
            trial_error = torch.zeros(5,device=device)
            for trial in range(no_trial):
                device_name = sim_params['device_name']
                input_bit = sim_params['input_bit']
                batch_interval = 1 + _crossbar_1.memristor_luts[device_name][
                    'total_no'] * _rows + 1 * input_bit  # reset + write + read
                _crossbar_1.batch_interval = batch_interval
                _crossbar_2.batch_interval = batch_interval
                _crossbar_3.batch_interval = batch_interval
                _crossbar_4.batch_interval = batch_interval

                # Perform c2c variation only
                sim_params['c2c_variation'] = True
                sim_params['d2d_variation'] = 0
                memristor_info_dict = _crossbar_1.memristor_info_dict
                memristor_info_dict[device_name]['sigma_relative'] = _var_rel
                memristor_info_dict[device_name]['sigma_absolute'] = _var_abs

                _crossbar_1.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_1.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_1.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_1.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                                         memristor_info_dict=memristor_info_dict)
        
                _crossbar_2.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_2.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_2.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_2.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                                         memristor_info_dict=memristor_info_dict)
        
                _crossbar_3.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_3.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_3.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_3.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                                         memristor_info_dict=memristor_info_dict)
        
                _crossbar_4.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_4.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_4.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_4.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar_1.to(device)
                _crossbar_1.set_batch_size_mimo(_batch_size)
        
                _crossbar_2.to(device)
                _crossbar_2.set_batch_size_mimo(_batch_size)
        
                _crossbar_3.to(device)
                _crossbar_3.set_batch_size_mimo(_batch_size)
        
                _crossbar_4.to(device)
                _crossbar_4.set_batch_size_mimo(_batch_size)
        
                # matrix and vector random generation
                # matrix = torch.rand(_rep, _rows, _cols, device=device)
                matrix_r = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
                matrix_i = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
                # matrix = torch.ones(_rep, _rows, _cols, device=device)
                # vector = torch.rand(_rep, 1, _rows, device=device)
                vector_r = -1 + 2 * torch.rand(_rep, read_batch, _rows, device=device)
                vector_i = -1 + 2 * torch.rand(_rep, read_batch, _rows, device=device)
                # vector = torch.ones(_rep, 1, _rows, device=device)
                # print("Randomized input")
        
                # Golden results calculation
                golden_model_1 = torch.matmul(vector_r, matrix_r)
                golden_model_2 = torch.matmul(vector_r, matrix_i)
                golden_model_3 = torch.matmul(vector_i, matrix_r)
                golden_model_4 = torch.matmul(vector_i, matrix_i)
                golden_r = golden_model_1 - golden_model_4
                golden_i = golden_model_2 + golden_model_3   
                    
                n_step = int(_rep / _batch_size)
                cross_1 = torch.zeros_like(golden_model_1, device=device)
                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix_r[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector_r[(step * _batch_size):(step * _batch_size + _batch_size)]
        
                    # Memristor-based results simulation
                    # Memristor crossbar program
                    _crossbar_1.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross_1[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_1.mapping_read_mimo(
                        target_v=vector_batch)
        
                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar_1.total_energy_calculation()
                        sim_power = _crossbar_1.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)
        
                    # mem_t update # Avoid mem_t at the last batch
                    if not step == n_step - 1:
                        _crossbar_1.mem_t_update()
        
                cross_2 = torch.zeros_like(golden_model_2, device=device)
                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix_i[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector_r[(step * _batch_size):(step * _batch_size + _batch_size)]
        
                    # Memristor-based results simulation
                    # Memristor crossbar program
                    _crossbar_2.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross_2[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_2.mapping_read_mimo(
                        target_v=vector_batch)
        
                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar_2.total_energy_calculation()
                        sim_power = _crossbar_2.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)
        
                    # mem_t update # Avoid mem_t at the last batch
                    if not step == n_step - 1:
                        _crossbar_2.mem_t_update()
        
                cross_3 = torch.zeros_like(golden_model_3, device=device)
                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix_r[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector_i[(step * _batch_size):(step * _batch_size + _batch_size)]
        
                    # Memristor-based results simulation
                    # Memristor crossbar program
                    _crossbar_3.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross_3[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_3.mapping_read_mimo(
                        target_v=vector_batch)
        
                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar_3.total_energy_calculation()
                        sim_power = _crossbar_3.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)
        
                    # mem_t update # Avoid mem_t at the last batch
                    if not step == n_step - 1:
                        _crossbar_3.mem_t_update()
        
                cross_4 = torch.zeros_like(golden_model_4, device=device)
                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix_i[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector_i[(step * _batch_size):(step * _batch_size + _batch_size)]
        
                    # Memristor-based results simulation
                    # Memristor crossbar program
                    _crossbar_4.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross_4[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_4.mapping_read_mimo(
                        target_v=vector_batch)
        
                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar_4.total_energy_calculation()
                        sim_power = _crossbar_4.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)
        
                    # mem_t update # Avoid mem_t at the last batch
                    if not step == n_step - 1:
                        _crossbar_4.mem_t_update()
        
                delta_cross_r = cross_1 - cross_4 - golden_r
                delta_cross_i = cross_2 + cross_3 - golden_i
                
                cross_error = torch.sqrt(
                    torch.sum(torch.square(delta_cross_r)) + torch.sum(torch.square(delta_cross_i))) / torch.sqrt(
                    torch.sum(torch.square(golden_r)) + torch.sum(torch.square(golden_i)))
        
                torch.set_printoptions(precision=8)
                trial_error[trial] = cross_error
                print("Absolute Sigma: ", _var_abs, ", Relative Sigma: ", _var_rel, ", Mean Error: ", cross_error)
            print("trial_mean_error:",torch.mean(trial_error))
    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)

def run_complex_sim(_crossbar_1, _crossbar_2, _crossbar_3, _crossbar_4, _rep, _batch_size, _rows, _cols,
                          sim_params, device,
                          _logs=[None, None, False, False, None], figs=None):
    print("<========================================>")
    print("Test case: ", _rep)
    file_name = "crossbar_size_test_case_r" + str(_rows) + "_c" + \
                str(_cols) + "_rep" + str(_rep) + ".csv"
    file_path = _logs[0]  # main file path
    header = ['size', 'AB_sigma', 'RE_sigma', 'me', 'mae', 'rmse', 'rmae', 'rrmse1', 'rrmse2', 'rpd1', 'rpd2', 'rpd3',
              'rpd4']
    file = file_path + "/" + file_name  # Location to the file for the main results
    # Only write header once
    if not (os.path.isfile(file)):
        utility.write_to_csv(file_path, file_name, header)

    print("<==============>")
    start_time = time.time()

    print("Row No. ", _rows, " Column No. ", _cols, " Repetition No. ", _rep, " Batch Size: ", _batch_size)

    print("<==============>")
    sigma_list = [0]
    _var_abs = 0
    _var_rel = 0
    no_trial = 10
    read_batch = 672
    trial_error = torch.zeros(no_trial,device=device)
    memristor_info_dict = _crossbar_1.memristor_info_dict
    _crossbar_1.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_1.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_1.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_1.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_1.shape,
                                             memristor_info_dict=memristor_info_dict)

    _crossbar_2.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_2.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_2.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_2.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_2.shape,
                                             memristor_info_dict=memristor_info_dict)

    _crossbar_3.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_3.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_3.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_3.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_3.shape,
                                             memristor_info_dict=memristor_info_dict)

    _crossbar_4.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_4.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_4.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_4.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar_4.shape,
                                             memristor_info_dict=memristor_info_dict)
    _crossbar_1.to(device)
    _crossbar_1.set_batch_size_mimo(_batch_size)

    _crossbar_2.to(device)
    _crossbar_2.set_batch_size_mimo(_batch_size)

    _crossbar_3.to(device)
    _crossbar_3.set_batch_size_mimo(_batch_size)

    _crossbar_4.to(device)
    _crossbar_4.set_batch_size_mimo(_batch_size)

    for trial in range(no_trial):
        device_name = sim_params['device_name']
        input_bit = sim_params['input_bit']
        batch_interval = 1 + _crossbar_1.memristor_luts[device_name][
            'total_no'] * _rows + read_batch * input_bit  # reset + write + read
        _crossbar_1.batch_interval = batch_interval

        # matrix and vector random generation      
        matrix_r = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
        matrix_i = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
        # matrix = torch.ones(_rep, _rows, _cols, device=device)
        # vector = torch.rand(_rep, 1, _rows, device=device)
        vector_r = -1 + 2 * torch.rand(_rep, read_batch, _rows, device=device)
        vector_i = -1 + 2 * torch.rand(_rep, read_batch, _rows, device=device)          
        # print("Randomized input")

        # Golden results calculation
        golden_model_1 = torch.matmul(vector_r, matrix_r)
        golden_model_2 = torch.matmul(vector_r, matrix_i)
        golden_model_3 = torch.matmul(vector_i, matrix_r)
        golden_model_4 = torch.matmul(vector_i, matrix_i)
        golden_r = golden_model_1 - golden_model_4
        golden_i = golden_model_2 + golden_model_3
            
        n_step = int(_rep / _batch_size)
        cross_1 = torch.zeros_like(golden_model_1, device=device)
        for step in range(n_step):
            # print(step)
            matrix_batch = matrix_r[(step * _batch_size):(step * _batch_size + _batch_size)]
            vector_batch = vector_r[(step * _batch_size):(step * _batch_size + _batch_size)]

            # Memristor-based results simulation
            # Memristor crossbar program
            _crossbar_1.mapping_write_mimo(target_x=matrix_batch)
            # Memristor crossbar perform matrix vector multiplication
            cross_1[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_1.mapping_read_mimo(
                target_v=vector_batch)

            if sim_params['hardware_estimation']:
                # print power results
                _crossbar_1.total_energy_calculation()
                sim_power = _crossbar_1.sim_power
                sim_periph_power = _crossbar_1.sim_periph_power
                total_energy = sim_power['total_energy'] + sim_periph_power['ADC_total_energy'] + sim_periph_power['DAC_total_energy']
                average_power = sim_power['average_power']
                
            # mem_t update # Avoid mem_t at the last batch
            if not step == n_step - 1:
                _crossbar_1.mem_t_update()

        cross_2 = torch.zeros_like(golden_model_2, device=device)
        for step in range(n_step):
            # print(step)
            matrix_batch = matrix_i[(step * _batch_size):(step * _batch_size + _batch_size)]
            vector_batch = vector_r[(step * _batch_size):(step * _batch_size + _batch_size)]

            # Memristor-based results simulation
            # Memristor crossbar program
            _crossbar_2.mapping_write_mimo(target_x=matrix_batch)
            # Memristor crossbar perform matrix vector multiplication
            cross_2[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_2.mapping_read_mimo(
                target_v=vector_batch)

            if sim_params['hardware_estimation']:
                # print power results
                _crossbar_2.total_energy_calculation()
                sim_power = _crossbar_2.sim_power
                sim_periph_power = _crossbar_2.sim_periph_power
                total_energy = sim_power['total_energy'] + sim_periph_power['ADC_total_energy'] + sim_periph_power['DAC_total_energy']
                average_power = sim_power['average_power']
                # print("total_energy=", total_energy)
                # print("average_power=", average_power)

            # mem_t update # Avoid mem_t at the last batch
            if not step == n_step - 1:
                _crossbar_2.mem_t_update()

        cross_3 = torch.zeros_like(golden_model_3, device=device)
        for step in range(n_step):
            # print(step)
            matrix_batch = matrix_r[(step * _batch_size):(step * _batch_size + _batch_size)]
            vector_batch = vector_i[(step * _batch_size):(step * _batch_size + _batch_size)]

            # Memristor-based results simulation
            # Memristor crossbar program
            _crossbar_3.mapping_write_mimo(target_x=matrix_batch)
            # Memristor crossbar perform matrix vector multiplication
            cross_3[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_3.mapping_read_mimo(
                target_v=vector_batch)

            if sim_params['hardware_estimation']:
                # print power results
                _crossbar_3.total_energy_calculation()
                sim_power = _crossbar_3.sim_power
                sim_periph_power = _crossbar_3.sim_periph_power
                total_energy = sim_power['total_energy'] + sim_periph_power['ADC_total_energy'] + sim_periph_power['DAC_total_energy']
                average_power = sim_power['average_power']
                # print("total_energy=", total_energy)
                # print("average_power=", average_power)

            # mem_t update # Avoid mem_t at the last batch
            if not step == n_step - 1:
                _crossbar_3.mem_t_update()

        cross_4 = torch.zeros_like(golden_model_4, device=device)
        for step in range(n_step):
            # print(step)
            matrix_batch = matrix_i[(step * _batch_size):(step * _batch_size + _batch_size)]
            vector_batch = vector_i[(step * _batch_size):(step * _batch_size + _batch_size)]

            # Memristor-based results simulation
            # Memristor crossbar program
            _crossbar_4.mapping_write_mimo(target_x=matrix_batch)
            # Memristor crossbar perform matrix vector multiplication
            cross_4[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_4.mapping_read_mimo(
                target_v=vector_batch)

            if sim_params['hardware_estimation']:
                # print power results
                _crossbar_4.total_energy_calculation()
                sim_power = _crossbar_4.sim_power
                sim_periph_power = _crossbar_4.sim_periph_power
                total_energy = sim_power['total_energy'] + sim_periph_power['ADC_total_energy'] + sim_periph_power['DAC_total_energy']
                average_power = sim_power['average_power']
                # print("total_energy=", total_energy)
                # print("average_power=", average_power)

            # mem_t update # Avoid mem_t at the last batch
            if not step == n_step - 1:
                _crossbar_4.mem_t_update()

        delta_cross_r = cross_1 - cross_4 - golden_r
        delta_cross_i = cross_2 + cross_3 - golden_i
        
        cross_error = torch.sqrt(
            torch.sum(torch.square(delta_cross_r)) + torch.sum(torch.square(delta_cross_i))) / torch.sqrt(
            torch.sum(torch.square(golden_r)) + torch.sum(torch.square(golden_i)))

        torch.set_printoptions(precision=8)
        print("cross_error=",cross_error)
        trial_error[trial] = cross_error
        
    print(torch.mean(trial_error))
    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)

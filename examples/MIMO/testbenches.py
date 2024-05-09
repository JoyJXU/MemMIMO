from utility import utility
import time
import os
import torch

from simbrain.memarray import MemristorArray


#############################################################
# Run a simulation of the crossbar based on the configuration
# def run_single_sim(_crossbar, _rep, _rows, _cols, _logs=[None, None, False, False, None]):
#     print("<========================================>")
#     print("Test case: ", _rep)
#     file_name = "test_case_r"+str(_rows)+"_c_" + \
#         str(_cols)+"_rep_"+str(_rep)+".csv"
#     file_path = _logs[0] #main file path
#     header = ['var_abs', 'var_rel']
#     for x in range(_cols):
#         header.append(str(x))
#     file = file_path+"/"+file_name # Location to the file for the main results
#     # Only write header once
#     if not (os.path.isfile(file)):
#         utility.write_to_csv(file_path, file_name, header)
#
#     print("<==============>")
#     start_time = time.time()
#     print("Row No. ", _rows, " Column No. ", _cols)
#
#     # matrix and vector random generation
#     matrix = torch.rand(_rows, _cols)
#     vector = -1 + 2 * torch.rand(_rows)
#     print("Randomized input")
#
#     # Golden results calculation
#     golden_model = torch.matmul(vector, matrix)
#
#     # Memristor-based results simulation
#     # Memristor crossbar program
#     _crossbar.mapping_write_mimo(target_x=matrix)
#     # Memristor crossbar perform matrix vector multiplication
#     cross = _crossbar.mapping_read_mimo(target_v=vector)
#
#     # Error calculation
#     error = utility.cal_error(golden_model, cross)
#     error = error.flatten(0, 2)
#
#     _var_abs = 0
#     _var_rel = 0
#     data = [str(_var_abs), str(_var_rel)]
#     [data.append(str(e.item())) for e in error]
#     utility.write_to_csv(file_path, file_name, data)
#
#     end_time = time.time()
#     exe_time = end_time - start_time
#     print("Execution time: ", exe_time)


# def run_c2c_sim(_crossbar, _rep, _batch_size, _rows, _cols, sim_params, device, _logs=[None, None, False, False, None]):
#     print("<========================================>")
#     print("Test case: ", _rep)
#     file_name = "c2c_test_case_r"+str(_rows)+"_c" + \
#         str(_cols)+"_rep"+str(_rep)+".csv"
#     file_path = _logs[0] #main file path
#     header = ['var_abs', 'var_rel']
#     for x in range(_cols):
#         header.append(str(x))
#     file = file_path+"/"+file_name # Location to the file for the main results
#     # Only write header once
#     if not (os.path.isfile(file)):
#         utility.write_to_csv(file_path, file_name, header)
#
#     print("<==============>")
#     start_time = time.time()
#     print("Row No. ", _rows, " Column No. ", _cols)
#
#     print("<==============>")
#     sigma_list = [0, 0.001, 0.01, 0.1, 1, 10]
#     print("Start Sigma: ", sigma_list[1], ", End Sigma: ", sigma_list[-1], ", Sigma=0 Included")
#
#     _var_abs = 0
#     _var_rel = 0
#     for _var_abs in sigma_list:
#         for _var_rel in sigma_list:
#             device_name = sim_params['device_name']
#             batch_interval = 1 + _crossbar.memristor_luts[device_name]['total_no'] + 1  # reset + write + read
#             _crossbar.batch_interval = batch_interval
#
#             # Perform c2c variation only
#             sim_params['c2c_variation'] = True
#             sim_params['d2d_variation'] = 0
#             memristor_info_dict = _crossbar.memristor_info_dict
#             memristor_info_dict[device_name]['sigma_relative'] = _var_rel
#             memristor_info_dict[device_name]['sigma_absolute'] = _var_abs
#             _crossbar.mem_array = MemristorArray(sim_params=sim_params, shape=_crossbar.shape, memristor_info_dict=memristor_info_dict)
#             _crossbar.to(device)
#             _crossbar.set_batch_size_mimo(_batch_size)
#
#             # matrix and vector random generation
#             matrix = torch.rand(_rep, _rows, _cols, device=device)
#             vector = -1 + 2 * torch.rand(_rep, 1, _rows, device=device)
#             # print("Randomized input")
#
#             # Golden results calculation
#             golden_model = torch.matmul(vector, matrix)
#
#             n_step = int(_rep / _batch_size)
#             cross = torch.zeros_like(golden_model, device=device)
#
#             for step in range(n_step):
#                 matrix_batch = matrix[(step * _batch_size):(step * _batch_size + _batch_size)]
#                 vector_batch = vector[(step * _batch_size):(step * _batch_size + _batch_size)]
#
#                 # Memristor-based results simulation
#                 # Memristor crossbar program
#                 _crossbar.mapping_write_mimo(target_x=matrix_batch)
#                 # Memristor crossbar perform matrix vector multiplication
#                 cross[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar.mapping_read_mimo(target_v=vector_batch)
#
#                 # mem_t update
#                 _crossbar.mem_t_update()
#
#             # Error calculation
#             error = utility.cal_error(golden_model, cross)
#             relative_error = error / golden_model
#             error = error.flatten(0, 2)
#             relative_error = relative_error.flatten(0, 2)
#
#             # data = [str(_var_abs), str(_var_rel)]
#             # [data.append(str(e.item())) for e in error]
#             # utility.write_to_csv(file_path, file_name, data)
#
#             me = torch.mean(error)
#             mae = torch.mean(abs(error))
#             rmse = torch.sqrt(torch.mean(error**2))
#             rmae = torch.mean(abs(relative_error))
#             rrmse = torch.sqrt(torch.mean(relative_error**2))
#             metrics = [me, mae, rmse, rmae, rrmse]
#
#             data = [str(_var_abs), str(_var_rel)]
#             [data.append(str(e.item())) for e in metrics]
#             utility.write_to_csv(file_path, file_name, data)
#
#             print("Absolute Sigma: ", _var_abs, ", Relative Sigma: ", _var_rel, ", Mean Error: ", me.item())
#
#     end_time = time.time()
#     exe_time = end_time - start_time
#     print("Execution time: ", exe_time)


# def run_signed_c2c_sim(_crossbar_pos, _crossbar_neg, _rep, _batch_size, _rows, _cols, sim_params, device,
#                        _logs=[None, None, False, False, None], figs=None):
#     print("<========================================>")
#     print("Test case: ", _rep)
#     file_name = "signed_c2c_test_case_r"+str(_rows)+"_c" + \
#         str(_cols)+"_rep"+str(_rep)+".csv"
#     file_path = _logs[0] #main file path
#     header = ['var_abs', 'var_rel']
#     for x in range(_cols):
#         header.append(str(x))
#     file = file_path+"/"+file_name # Location to the file for the main results
#     # Only write header once
#     if not (os.path.isfile(file)):
#         utility.write_to_csv(file_path, file_name, header)
#
#     print("<==============>")
#     start_time = time.time()
#     print("Row No. ", _rows, " Column No. ", _cols)
#
#     print("<==============>")
#     sigma_list = [0, 0.001, 0.01, 0.1, 1, 10]
#     print("Start Sigma: ", sigma_list[1], ", End Sigma: ", sigma_list[-1], ", Sigma=0 Included")
#
#     _var_abs = 0
#     _var_rel = 0
#     for _var_abs in sigma_list:
#         for _var_rel in sigma_list:
#             device_name = sim_params['device_name']
#             batch_interval = 1 + _crossbar_pos.memristor_luts[device_name]['total_no'] + 1  # reset + write + read
#             _crossbar_pos.batch_interval = batch_interval
#             _crossbar_neg.batch_interval = batch_interval
#
#             # Perform c2c variation only
#             sim_params['c2c_variation'] = True
#             sim_params['d2d_variation'] = 0
#             memristor_info_dict = _crossbar_pos.memristor_info_dict
#             memristor_info_dict[device_name]['sigma_relative'] = _var_rel
#             memristor_info_dict[device_name]['sigma_absolute'] = _var_abs
#             _crossbar_pos.mem_array = MemristorArray(sim_params=sim_params, shape=_crossbar_pos.shape,
#                                                      memristor_info_dict=memristor_info_dict)
#             _crossbar_neg.mem_array = MemristorArray(sim_params=sim_params, shape=_crossbar_neg.shape,
#                                                      memristor_info_dict=memristor_info_dict)
#             _crossbar_pos.to(device)
#             _crossbar_neg.to(device)
#             _crossbar_pos.set_batch_size_mimo(_batch_size)
#             _crossbar_neg.set_batch_size_mimo(_batch_size)
#
#             # matrix and vector random generation
#             # matrix = torch.rand(_rep, _rows, _cols, device=device)
#             matrix = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
#             # matrix = torch.ones(_rep, _rows, _cols, device=device)
#             # vector = torch.rand(_rep, 1, _rows, device=device)
#             vector = -1 + 2 * torch.rand(_rep, 1, _rows, device=device)
#             # vector = torch.ones(_rep, 1, _rows, device=device)
#             # print("Randomized input")
#
#             # Golden results calculation
#             golden_model = torch.matmul(vector, matrix)
#
#             n_step = int(_rep / _batch_size)
#             cross = torch.zeros_like(golden_model, device=device)
#             cross_pos = torch.zeros_like(golden_model, device=device)
#             cross_neg = torch.zeros_like(golden_model, device=device)
#
#             for step in range(n_step):
#                 matrix_batch = matrix[(step * _batch_size):(step * _batch_size + _batch_size)]
#                 vector_batch = vector[(step * _batch_size):(step * _batch_size + _batch_size)]
#
#                 # Enable signed matrix
#                 matrix_pos = torch.relu(matrix_batch)
#                 matrix_neg = torch.relu(matrix_batch * -1)
#
#                 # Memristor-based results simulation
#                 # Memristor crossbar program
#                 _crossbar_pos.mapping_write_mimo(target_x=matrix_pos)
#                 _crossbar_neg.mapping_write_mimo(target_x=matrix_neg)
#                 # Memristor crossbar perform matrix vector multiplication
#                 cross_pos[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_pos.mapping_read_mimo(
#                     target_v=vector_batch)
#                 cross_neg[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar_neg.mapping_read_mimo(
#                     target_v=(vector_batch * -1))
#                 cross = cross_pos + cross_neg
#
#                 # mem_t update
#                 _crossbar_pos.mem_t_update()
#                 _crossbar_neg.mem_t_update()
#
#             # Error calculation
#             error = utility.cal_error(golden_model, cross)
#             relative_error = error / golden_model
#             error = error.flatten(0, 2)
#             relative_error = relative_error.flatten(0, 2)
#
#             utility.plot_distribution(figs, vector, matrix, golden_model, cross, error, relative_error)
#
#             # data = [str(_var_abs), str(_var_rel)]
#             # [data.append(str(e.item())) for e in error]
#             # utility.write_to_csv(file_path, file_name, data)
#
#             me = torch.mean(error)
#             mae = torch.mean(abs(error))
#             rmse = torch.sqrt(torch.mean(error**2))
#             rmae = torch.mean(abs(relative_error))
#             rrmse = torch.sqrt(torch.mean(relative_error**2))
#             metrics = [me, mae, rmse, rmae, rrmse]
#
#             data = [str(_var_abs), str(_var_rel)]
#             [data.append(str(e.item())) for e in metrics]
#             utility.write_to_csv(file_path, file_name, data)
#
#             print("Absolute Sigma: ", _var_abs, ", Relative Sigma: ", _var_rel, ", Mean Error: ", me.item())
#
#     end_time = time.time()
#     exe_time = end_time - start_time
#     print("Execution time: ", exe_time)


def run_d2d_sim(_crossbar, _rep, _batch_size, _rows, _cols, sim_params, device, _logs=[None, None, False, False, None]):
    print("<========================================>")
    print("Test case: ", _rep)
    file_name = "d2d_test_case_r" + str(_rows) + "_c" + str(_cols) + "_rep" + str(_rep) + ".csv"
    file_path = _logs[0] #main file path
    header = ['size', 'G_sigma', 'lin_sigma', 'me', 'mae', 'rmse', 'rmae', 'rrmse1', 'rrmse2', 'rpd1', 'rpd2', 'rpd3', 'rpd4']
    file = file_path + "/" + file_name  # Location to the file for the main results
    # Only write header once
    if not (os.path.isfile(file)):
        utility.write_to_csv(file_path, file_name, header)

    print("<==============>")
    start_time = time.time()
    print("Row No. ", _rows, " Column No. ", _cols, " Repetition No. ", _rep, " Batch Size: ", _batch_size)

    print("<==============>")
    sigma_list = [0, 0.001, 0.01, 0.1, 1, 10]
    print("Start Sigma: ", sigma_list[1], ", End Sigma: ", sigma_list[-1], ", Sigma=0 Included")

    _var_g = 0
    _var_linearity = 0
    no_trial = 5
    for _var_g in sigma_list:
        for _var_linearity in sigma_list:
            for trial in range(no_trial):
                device_name = sim_params['device_name']
                input_bit = sim_params['input_bit']
                batch_interval = 1 + _crossbar.memristor_luts[device_name]['total_no'] * _rows + 1 * input_bit # reset + write + read
                _crossbar.batch_interval = batch_interval

                # Perform d2d variation only
                sim_params['c2c_variation'] = False
                sim_params['d2d_variation'] = 1
                memristor_info_dict = _crossbar.memristor_info_dict
                G_off = memristor_info_dict[device_name]['G_off']
                G_on = memristor_info_dict[device_name]['G_on']
                memristor_info_dict[device_name]['Gon_sigma'] = G_on * _var_g
                memristor_info_dict[device_name]['Goff_sigma'] = G_off * _var_g

                P_off = memristor_info_dict[device_name]['P_off']
                P_on = memristor_info_dict[device_name]['P_on']
                memristor_info_dict[device_name]['Pon_sigma'] = P_on * _var_linearity
                memristor_info_dict[device_name]['Poff_sigma'] = P_off * _var_linearity

                _crossbar.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                                       memristor_info_dict=memristor_info_dict)
                _crossbar.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                                       memristor_info_dict=memristor_info_dict)
                _crossbar.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                                       memristor_info_dict=memristor_info_dict)
                _crossbar.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                                       memristor_info_dict=memristor_info_dict)
                _crossbar.to(device)
                _crossbar.set_batch_size_mimo(_batch_size)

                # matrix and vector random generation
                # matrix = torch.rand(_rep, _rows, _cols, device=device)
                matrix = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
                # matrix = torch.ones(_rep, _rows, _cols, device=device)
                # vector = torch.rand(_rep, 1, _rows, device=device)
                vector = -1 + 2 * torch.rand(_rep, 1, _rows, device=device)
                # vector = torch.ones(_rep, 1, _rows, device=device)
                # print("Randomized input")

                # Golden results calculation
                golden_model = torch.matmul(vector, matrix)

                n_step = int(_rep / _batch_size)
                cross = torch.zeros_like(golden_model, device=device)

                for step in range(n_step):
                    matrix_batch = matrix[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector[(step * _batch_size):(step * _batch_size + _batch_size)]

                    # Memristor-based results simulation
                    if sim_params['stuck_at_fault'] == True:
                        _crossbar.update_SAF_mask()
                    # Memristor crossbar program
                    _crossbar.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar.mapping_read_mimo(target_v=vector_batch)

                    # mem_t update
                    _crossbar.mem_t_update()

                    if sim_params['hardware_estimation']:
                        # print power results
                        _crossbar.total_energy_calculation()
                        sim_power = _crossbar.sim_power
                        total_energy = sim_power['total_energy']
                        average_power = sim_power['average_power']
                        print("total_energy=", total_energy)
                        print("average_power=", average_power)

                # Error calculation
                error = utility.cal_error(golden_model, cross)
                relative_error = error / golden_model
                rpd1_error = 2 * abs(error / (torch.abs(golden_model) + torch.abs(cross)))
                rpd2_error = abs(error / torch.max(torch.abs(golden_model), torch.abs(cross)))
                rpd3_error = error / (torch.abs(golden_model) + 0.001)
                rpd4_error = error / (torch.abs(golden_model) + 1)

                error = error.flatten(0, 2)
                relative_error = relative_error.flatten(0, 2)
                rpd1_error = rpd1_error.flatten(0, 2)
                rpd2_error = rpd2_error.flatten(0, 2)
                rpd3_error = rpd3_error.flatten(0, 2)
                rpd4_error = rpd4_error.flatten(0, 2)
                print('Error Calculation Done')
                print("<==============>")

                me = torch.mean(error)
                mae = torch.mean(abs(error))
                rmse = torch.sqrt(torch.mean(error ** 2))
                rmae = torch.mean(abs(relative_error))
                rrmse1 = torch.sqrt(torch.mean(relative_error ** 2))
                rrmse2 = torch.sqrt(torch.sum(error ** 2) / torch.sum(golden_model.flatten(0, 2) ** 2))
                rpd1 = torch.mean(rpd1_error)
                rpd2 = torch.mean(rpd2_error)
                rpd3 = torch.mean(abs(rpd3_error))
                rpd4 = torch.mean(abs(rpd4_error))
                metrics = [me, mae, rmse, rmae, rrmse1, rrmse2, rpd1, rpd2, rpd3, rpd4]

                data = [str(_rows), str(_var_g), str(_var_linearity)]
                [data.append(str(e.item())) for e in metrics]
                utility.write_to_csv(file_path, file_name, data)

                print("Gon/Goff Sigma: ", _var_g, ", Nonlinearity Sigma: ", _var_linearity, ", Mean Error: ", me.item())

    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)


def run_c2c_sim(_crossbar, _rep, _batch_size, _rows, _cols, sim_params, device,
                              _logs=[None, None, False, False, None], figs=None):
    print("<========================================>")
    print("Test case: ", _rep)
    file_name = "crossbar_size_test_case_r"+str(_rows)+"_c" + \
        str(_cols)+"_rep"+str(_rep)+".csv"
    file_path = _logs[0] #main file path
    header = ['size', 'AB_sigma', 'RE_sigma', 'me', 'mae', 'rmse', 'rmae', 'rrmse1', 'rrmse2', 'rpd1', 'rpd2', 'rpd3', 'rpd4']
    file = file_path+"/"+file_name # Location to the file for the main results
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
    sigma_list = [0]
    _var_abs = 0
    _var_rel = 0
    no_trial = 5
    for _var_abs in sigma_list:
        for _var_rel in sigma_list:
            for trial in range(no_trial):
                device_name = sim_params['device_name']
                input_bit = sim_params['input_bit']
                batch_interval = 1 + _crossbar.memristor_luts[device_name]['total_no'] * _rows + 1 * input_bit  # reset + write + read
                _crossbar.batch_interval = batch_interval

                # Perform c2c variation only
                sim_params['c2c_variation'] = True
                sim_params['d2d_variation'] = 0
                memristor_info_dict = _crossbar.memristor_info_dict
                memristor_info_dict[device_name]['sigma_relative'] = _var_rel
                memristor_info_dict[device_name]['sigma_absolute'] = _var_abs
                _crossbar.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                                         memristor_info_dict=memristor_info_dict)
                _crossbar.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                                       memristor_info_dict=memristor_info_dict)
                _crossbar.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                                       memristor_info_dict=memristor_info_dict)
                _crossbar.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                                       memristor_info_dict=memristor_info_dict)
                _crossbar.to(device)
                _crossbar.set_batch_size_mimo(_batch_size)

                # matrix and vector random generation
                # matrix = torch.rand(_rep, _rows, _cols, device=device)
                matrix = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
                # matrix = torch.ones(_rep, _rows, _cols, device=device)
                # vector = torch.rand(_rep, 1, _rows, device=device)
                vector = -1 + 2 * torch.rand(_rep, 1, _rows, device=device)
                # vector = torch.ones(_rep, 1, _rows, device=device)
                # print("Randomized input")

                # Golden results calculation
                golden_model = torch.matmul(vector, matrix)

                n_step = int(_rep / _batch_size)
                cross = torch.zeros_like(golden_model, device=device)

                for step in range(n_step):
                    # print(step)
                    matrix_batch = matrix[(step * _batch_size):(step * _batch_size + _batch_size)]
                    vector_batch = vector[(step * _batch_size):(step * _batch_size + _batch_size)]

                    # Memristor-based results simulation
                    if sim_params['stuck_at_fault'] == True:
                        _crossbar.update_SAF_mask()
                    # Memristor crossbar program
                    _crossbar.mapping_write_mimo(target_x=matrix_batch)
                    # Memristor crossbar perform matrix vector multiplication
                    cross[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar.mapping_read_mimo(
                        target_v=vector_batch)

                    # mem_t update
                    _crossbar.mem_t_update()

                    # print power results
                    _crossbar.total_energy_calculation()
                    sim_power = _crossbar.sim_power
                    total_energy = sim_power['total_energy']
                    average_power = sim_power['average_power']
                    print("total_energy=", total_energy)
                    print("average_power=", average_power)

                # Error calculation
                error = utility.cal_error(golden_model, cross)
                relative_error = error / golden_model
                rpd1_error = 2 * abs(error / (torch.abs(golden_model) + torch.abs(cross)))
                rpd2_error = abs(error / torch.max(torch.abs(golden_model), torch.abs(cross)))
                rpd3_error = error / (torch.abs(golden_model) + 0.001)
                rpd4_error = error / (torch.abs(golden_model) + 1)

                error = error.flatten(0, 2)
                relative_error = relative_error.flatten(0, 2)
                rpd1_error = rpd1_error.flatten(0, 2)
                rpd2_error = rpd2_error.flatten(0, 2)
                rpd3_error = rpd3_error.flatten(0, 2)
                rpd4_error = rpd4_error.flatten(0, 2)
                print('Error Calculation Done')
                print("<==============>")

                utility.plot_distribution(figs, vector, matrix, golden_model, cross, error, relative_error, rpd1_error, rpd2_error, rpd3_error, rpd4_error)
                print('Visualization Done')
                print("<==============>")

                # data = [str(_var_abs), str(_var_rel)]
                # [data.append(str(e.item())) for e in error]
                # utility.write_to_csv(file_path, file_name, data)

                me = torch.mean(error)
                mae = torch.mean(abs(error))
                rmse = torch.sqrt(torch.mean(error**2))
                rmae = torch.mean(abs(relative_error))
                rrmse1 = torch.sqrt(torch.mean(relative_error**2))
                rrmse2 = torch.sqrt(torch.sum(error ** 2) / torch.sum(golden_model.flatten(0, 2) ** 2))
                rpd1 = torch.mean(rpd1_error)
                rpd2 = torch.mean(rpd2_error)
                rpd3 = torch.mean(abs(rpd3_error))
                rpd4 = torch.mean(abs(rpd4_error))
                metrics = [me, mae, rmse, rmae, rrmse1, rrmse2, rpd1, rpd2, rpd3, rpd4]


                data = [str(_rows), str(_var_abs), str(_var_rel)]
                [data.append(str(e.item())) for e in metrics]
                utility.write_to_csv(file_path, file_name, data)

                print("Absolute Sigma: ", _var_abs, ", Relative Sigma: ", _var_rel, ", Mean Error: ", me.item())

    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)


def run_crossbar_size_sim(_crossbar, _rep, _batch_size, _rows, _cols, sim_params, device,
                              _logs=[None, None, False, False, None], figs=None):
    print("<========================================>")
    print("Test case: ", _rep)
    file_name = "crossbar_size_test_case_r"+str(_rows)+"_c" + \
        str(_cols)+"_rep"+str(_rep)+".csv"
    file_path = _logs[0] #main file path
    header = ['size', 'AB_sigma', 'RE_sigma', 'me', 'mae', 'rmse', 'rmae', 'rrmse1', 'rrmse2', 'rpd1', 'rpd2', 'rpd3', 'rpd4']
    file = file_path+"/"+file_name # Location to the file for the main results
    # Only write header once
    if not (os.path.isfile(file)):
        utility.write_to_csv(file_path, file_name, header)

    print("<==============>")
    start_time = time.time()

    # # Batch Size Adaption
    # if (_batch_size * _rows) > 2e6 and _batch_size >= 10:
    #     _batch_size = int(_batch_size / 10)
    # elif (_batch_size * _rows) < 2e5 and _batch_size <= (_rep / 10):
    #     _batch_size = int(_batch_size * 10)

    print("Row No. ", _rows, " Column No. ", _cols, " Repetition No. ", _rep, " Batch Size: ", _batch_size)

    print("<==============>")
    sigma_list = [0]
    _var_abs = 0
    _var_rel = 0
    no_trial = 5
    read_batch = 1

    for trial in range(no_trial):
        device_name = sim_params['device_name']
        input_bit = sim_params['input_bit']
        batch_interval = 1 + _crossbar.memristor_luts[device_name]['total_no'] * _rows + read_batch * input_bit  # reset + write + read
        _crossbar.batch_interval = batch_interval

        _var_g = 0.055210197891837
        _var_linearity = 0.1
        sim_params['d2d_variation'] = 1
        memristor_info_dict = _crossbar.memristor_info_dict
        G_off = memristor_info_dict[device_name]['G_off']
        G_on = memristor_info_dict[device_name]['G_on']
        memristor_info_dict[device_name]['Gon_sigma'] = G_on * _var_g
        memristor_info_dict[device_name]['Goff_sigma'] = G_off * _var_g

        P_off = memristor_info_dict[device_name]['P_off']
        P_on = memristor_info_dict[device_name]['P_on']
        memristor_info_dict[device_name]['Pon_sigma'] = P_on * _var_linearity
        memristor_info_dict[device_name]['Poff_sigma'] = P_off * _var_linearity

        _crossbar.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                               memristor_info_dict=memristor_info_dict)
        _crossbar.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                               memristor_info_dict=memristor_info_dict)
        _crossbar.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                               memristor_info_dict=memristor_info_dict)
        _crossbar.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=_crossbar.shape,
                                               memristor_info_dict=memristor_info_dict)

        _crossbar.to(device)
        _crossbar.set_batch_size_mimo(_batch_size)

        # matrix and vector random generation
        # matrix = torch.rand(_rep, _rows, _cols, device=device)
        matrix = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
        # matrix = torch.ones(_rep, _rows, _cols, device=device)
        # vector = torch.rand(_rep, 1, _rows, device=device)
        vector = -1 + 2 * torch.rand(_rep, read_batch, _rows, device=device)
        # vector = torch.ones(_rep, 1, _rows, device=device)
        # print("Randomized input")

        # Golden results calculation
        golden_model = torch.matmul(vector, matrix)

        n_step = int(_rep / _batch_size)
        cross = torch.zeros_like(golden_model, device=device)

        for step in range(n_step):
            # print(step)
            matrix_batch = matrix[(step * _batch_size):(step * _batch_size + _batch_size)]
            vector_batch = vector[(step * _batch_size):(step * _batch_size + _batch_size)]

            # Memristor-based results simulation
            if sim_params['stuck_at_fault'] == True:
                _crossbar.update_SAF_mask()
            # Memristor crossbar program
            _crossbar.mapping_write_mimo(target_x=matrix_batch)
            # Memristor crossbar perform matrix vector multiplication
            cross[(step * _batch_size):(step * _batch_size + _batch_size)] = _crossbar.mapping_read_mimo(
                target_v=vector_batch)

            if sim_params['hardware_estimation']:
                # print power results
                _crossbar.total_energy_calculation()
                sim_power = _crossbar.sim_power
                total_energy = sim_power['total_energy']
                average_power = sim_power['average_power']
                print("total_energy=", total_energy)
                print("average_power=", average_power)

            # mem_t update # Avoid mem_t at the last batch
            if not step == n_step - 1:
                _crossbar.mem_t_update()

        # Error calculation
        error = utility.cal_error(golden_model, cross)
        relative_error = error / golden_model
        rpd1_error = 2 * abs(error / (torch.abs(golden_model) + torch.abs(cross)))
        rpd2_error = abs(error / torch.max(torch.abs(golden_model), torch.abs(cross)))
        rpd3_error = error / (torch.abs(golden_model) + 0.001)
        rpd4_error = error / (torch.abs(golden_model) + 1)

        error = error.flatten(0, 2)
        relative_error = relative_error.flatten(0, 2)
        rpd1_error = rpd1_error.flatten(0, 2)
        rpd2_error = rpd2_error.flatten(0, 2)
        rpd3_error = rpd3_error.flatten(0, 2)
        rpd4_error = rpd4_error.flatten(0, 2)
        print('Error Calculation Done')
        print("<==============>")

        utility.plot_distribution(figs, vector, matrix, golden_model, cross, error, relative_error, rpd1_error, rpd2_error, rpd3_error, rpd4_error)
        print('Visualization Done')
        print("<==============>")

        # data = [str(_var_abs), str(_var_rel)]
        # [data.append(str(e.item())) for e in error]
        # utility.write_to_csv(file_path, file_name, data)

        me = torch.mean(error)
        mae = torch.mean(abs(error))
        rmse = torch.sqrt(torch.mean(error**2))
        rmae = torch.mean(abs(relative_error))
        rrmse1 = torch.sqrt(torch.mean(relative_error**2))
        rrmse2 = torch.sqrt(torch.sum(error ** 2) / torch.sum(golden_model.flatten(0, 2) ** 2))
        rpd1 = torch.mean(rpd1_error)
        rpd2 = torch.mean(rpd2_error)
        rpd3 = torch.mean(abs(rpd3_error))
        rpd4 = torch.mean(abs(rpd4_error))
        metrics = [me, mae, rmse, rmae, rrmse1, rrmse2, rpd1, rpd2, rpd3, rpd4]


        data = [str(_rows), str(_var_abs), str(_var_rel)]
        [data.append(str(e.item())) for e in metrics]
        utility.write_to_csv(file_path, file_name, data)

        print("Absolute Sigma: ", _var_abs, ", Relative Sigma: ", _var_rel, ", Mean Error: ", me.item())

    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)
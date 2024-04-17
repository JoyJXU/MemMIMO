#!python3

"""
MIT License

Copyright (c) 2023 Dimitrios Stathis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Crossbar dimensions:
# Rows --> Input sources
# Columns --> Outputs during computation, ground during programming

# Calculation of Voltages depending on the state of the devices (R) and the Voltage sources

from utility import utility
import time
import os
import torch
import matplotlib.pyplot as plt
import sys
import json

sys.path.append("../")


from simbrain.mapping import MimoMapping

from simbrain.memarray import MemristorArray


def run_crossbar_size_sim(
    _crossbar,
    _rep,
    _batch_size,
    _rows,
    _cols,
    sim_params,
    device,
    _logs=[None, None, False, False, None],
    figs=None,
):
    print("<========================================>")
    print("Test case: ", _rep)
    file_name = (
        "crossbar_size_test_case_r"
        + str(_rows)
        + "_c"
        + str(_cols)
        + "_rep"
        + str(_rep)
        + ".csv"
    )
    file_path = _logs[0]  # main file path
    header = [
        "size",
        "AB_sigma",
        "RE_sigma",
        "me",
        "mae",
        "rmse",
        "rmae",
        "rrmse1",
        "rrmse2",
        "rpd1",
        "rpd2",
        "rpd3",
        "rpd4",
    ]
    file = file_path + "/" + file_name  # Location to the file for the main results
    # Only write header once
    if not (os.path.isfile(file)):
        utility.write_to_csv(file_path, file_name, header)

    print("<==============>")
    start_time = time.time()

    print(
        "Row No. ",
        _rows,
        " Column No. ",
        _cols,
        " Repetition No. ",
        _rep,
        " Batch Size: ",
        _batch_size,
    )

    print("<==============>")
    sigma_list = [0]
    _var_abs = 0
    _var_rel = 0
    no_trial = 5
    read_batch = 1

    for trial in range(no_trial):
        device_name = sim_params["device_name"]
        input_bit = sim_params["input_bit"]
        batch_interval = (
            1
            + _crossbar.memristor_luts[device_name]["total_no"] * _rows
            + read_batch * input_bit
        )  # reset + write + read
        _crossbar.batch_interval = batch_interval

        _var_g = 0.055210197891837
        _var_linearity = 0.1
        sim_params["d2d_variation"] = 1
        memristor_info_dict = _crossbar.memristor_info_dict
        G_off = memristor_info_dict[device_name]["G_off"]
        G_on = memristor_info_dict[device_name]["G_on"]
        memristor_info_dict[device_name]["Gon_sigma"] = G_on * _var_g
        memristor_info_dict[device_name]["Goff_sigma"] = G_off * _var_g

        P_off = memristor_info_dict[device_name]["P_off"]
        P_on = memristor_info_dict[device_name]["P_on"]
        memristor_info_dict[device_name]["Pon_sigma"] = P_on * _var_linearity
        memristor_info_dict[device_name]["Poff_sigma"] = P_off * _var_linearity

        _crossbar.mem_pos_pos = MemristorArray(
            sim_params=sim_params,
            shape=_crossbar.shape,
            memristor_info_dict=memristor_info_dict,
        )
        _crossbar.mem_neg_pos = MemristorArray(
            sim_params=sim_params,
            shape=_crossbar.shape,
            memristor_info_dict=memristor_info_dict,
        )
        _crossbar.mem_pos_neg = MemristorArray(
            sim_params=sim_params,
            shape=_crossbar.shape,
            memristor_info_dict=memristor_info_dict,
        )
        _crossbar.mem_neg_neg = MemristorArray(
            sim_params=sim_params,
            shape=_crossbar.shape,
            memristor_info_dict=memristor_info_dict,
        )

        _crossbar.to(device)
        _crossbar.set_batch_size_mimo(_batch_size)

        matrix = -1 + 2 * torch.rand(_rep, _rows, _cols, device=device)
        vector = -1 + 2 * torch.rand(_rep, read_batch, _rows, device=device)

        # Golden results calculation
        golden_model = torch.matmul(vector, matrix)

        n_step = int(_rep / _batch_size)
        cross = torch.zeros_like(golden_model, device=device)

        for step in range(n_step):
            # print(step)
            matrix_batch = matrix[
                (step * _batch_size) : (step * _batch_size + _batch_size)
            ]
            vector_batch = vector[
                (step * _batch_size) : (step * _batch_size + _batch_size)
            ]

            # Memristor-based results simulation
            # Memristor crossbar program
            _crossbar.mapping_write_mimo(target_x=matrix_batch)
            # Memristor crossbar perform matrix vector multiplication
            cross[(step * _batch_size) : (step * _batch_size + _batch_size)] = (
                _crossbar.mapping_read_mimo(target_v=vector_batch)
            )

            if sim_params["power_estimation"]:
                # print power results
                _crossbar.total_energy_calculation()
                sim_power = _crossbar.sim_power
                total_energy = sim_power["total_energy"]
                average_power = sim_power["average_power"]
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
        print("Error Calculation Done")
        print("<==============>")

        utility.plot_distribution(
            figs,
            vector,
            matrix,
            golden_model,
            cross,
            error,
            relative_error,
            rpd1_error,
            rpd2_error,
            rpd3_error,
            rpd4_error,
        )
        print("Visualization Done")
        print("<==============>")

        me = torch.mean(error)
        mae = torch.mean(abs(error))
        rmse = torch.sqrt(torch.mean(error**2))
        rmae = torch.mean(abs(relative_error))
        rrmse1 = torch.sqrt(torch.mean(relative_error**2))
        rrmse2 = torch.sqrt(
            torch.sum(error**2) / torch.sum(golden_model.flatten(0, 2) ** 2)
        )
        rpd1 = torch.mean(rpd1_error)
        rpd2 = torch.mean(rpd2_error)
        rpd3 = torch.mean(abs(rpd3_error))
        rpd4 = torch.mean(abs(rpd4_error))
        metrics = [me, mae, rmse, rmae, rrmse1, rrmse2, rpd1, rpd2, rpd3, rpd4]

        data = [str(_rows), str(_var_abs), str(_var_rel)]
        [data.append(str(e.item())) for e in metrics]
        utility.write_to_csv(file_path, file_name, data)

        print(
            "Absolute Sigma: ",
            _var_abs,
            ", Relative Sigma: ",
            _var_rel,
            ", Mean Error: ",
            me.item(),
        )

    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)


#############################################################
# parser = argparse.ArgumentParser()
# parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--gpu", dest="gpu", action="store_true", default="gpu")
# parser.add_argument("--rows", type=int, default=8)
# parser.add_argument("--cols", type=int, default=1)
# parser.add_argument("--rep", type=int, default=10000)
# parser.add_argument("--batch_size", type=int, default=1000)
# parser.add_argument(
#    "--memristor_structure", type=str, default="mimo"
# )  # trace, mimo or crossbar
# parser.add_argument(
#    "--memristor_device", type=str, default="new_ferro"
# )  # ideal, ferro, or hu
# parser.add_argument("--c2c_variation", type=bool, default=False)
# parser.add_argument(
#    "--d2d_variation", type=int, default=0
# )  # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
# parser.add_argument("--stuck_at_fault", type=bool, default=False)
# parser.add_argument(
#    "--retention_loss", type=int, default=0
# )  # retention loss, 0: without it, 1: during pulse, 2: no pluse for a long time
# parser.add_argument(
#    "--aging_effect", type=int, default=0
# )  # 0: No aging effect, 1: equation 1, 2: equation 2
# parser.add_argument(
#    "--process_node", type=int, default=200
# )  # In practice, process_node shall be set around 1/2 of the memristor size; Hu: 10um; Ferro:200nm;
# parser.add_argument("--input_bit", type=int, default=8)
# parser.add_argument("--power_estimation", type=int, default=True)
# args = parser.parse_args()


def parse_json(json_file):
    # Set default values
    default_config = {
        "seed": 0,
        "gpu": False,
        "rows": 8,
        "cols": 1,
        "rep": 10000,
        "batch_size": 1000,
        "memristor_structure": "mimo",
        "memristor_device": "new_ferro",
        "c2c_variation": False,
        "d2d_variation": 0,
        "stuck_at_fault": False,
        "retention_loss": 0,
        "aging_effect": 0,
        "process_node": 200,
        "input_bit": 8,
        "power_estimation": True,
    }

    # Open and read the JSON file
    try:
        with open(json_file, "r") as file:
            config_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {json_file} does not exist.")
        return default_config
    except json.JSONDecodeError:
        print("Error: JSON decoding failed.")
        return default_config

    # Update the default config with values from the file
    for key in default_config:
        config_data.setdefault(key, default_config[key])

    return config_data


def single_run():
    args = parse_json("..\single_run_config.json")
    return args
    # seed = args.seed # Fixe Seed
    seed = int(time.time())  # Random Seed
    gpu = args.gpu
    # Sets up Gpu use
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, [1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(seed)
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        device = "cpu"
        if gpu:
            gpu = False

    torch.set_num_threads(os.cpu_count() - 1)
    print("Running on Device = ", device)

    _rows = args.rows
    _cols = args.cols
    _rep = args.rep
    _batch_size = args.batch_size
    _logs = ["test_data", None, False, False, None]

    mem_device = {
        "device_structure": args.memristor_structure,
        "device_name": args.memristor_device,
        "c2c_variation": args.c2c_variation,
        "d2d_variation": args.d2d_variation,
        "stuck_at_fault": args.stuck_at_fault,
        "retention_loss": args.retention_loss,
        "aging_effect": args.aging_effect,
        "process_node": args.process_node,
        "input_bit": args.input_bit,
        "batch_interval": 1,
        "power_estimation": args.power_estimation,
    }

    # Run crossbar size experiments
    # size_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    size_list = [16]
    # size_list = [2048, 256]
    for _rows in size_list:
        _crossbar = MimoMapping(sim_params=mem_device, shape=(_rows, _cols))
        _crossbar.to(device)

        # Area print
        _crossbar.total_area_calculation()
        print("total crossbar area=", _crossbar.sim_area["mem_area"], " m2")

        # run_d2d_sim(_crossbar, _rep, _batch_size, _rows, _cols, mem_device, device, _logs)
        run_crossbar_size_sim(
            _crossbar, _rep, _batch_size, _rows, _cols, mem_device, device, _logs
        )


def main():
    print("Single run experiment!")
    print(single_run())


if __name__ == "__main__":
    main()

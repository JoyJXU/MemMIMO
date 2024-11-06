import argparse
import sys
import torch
import time
import os

SIMBRAIN_PATH = os.getenv("SIMBRAIN_PATH")

sys.path.append(SIMBRAIN_PATH)

from testbenches import *
from simbrain.mapping import MimoMapping

#############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rows", type=int, default=16)
parser.add_argument("--cols", type=int, default=64)
parser.add_argument("--rep", type=int, default=1)
parser.add_argument("--read_batch", type=int, default=672) # correspond to operations / update ratio
parser.add_argument("--write_batch_size", type=int, default=1)
parser.add_argument("--memristor_structure", type=str, default='mimo')
parser.add_argument("--memristor_device", type=str, default='MF') # ideal, ferro, hu(FS) or MF
parser.add_argument("--c2c_variation", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--d2d_variation", type=int, default=0) # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
parser.add_argument("--input_bit", type=int, default=8)
parser.add_argument("--ADC_precision", type=int, default=16)
parser.add_argument("--ADC_setting", type=int, default=4)  # 2:two memristor crossbars use one ADC; 4:one memristor crossbar use one ADC
parser.add_argument("--ADC_rounding_function", type=str, default='floor')  # floor or round
parser.add_argument("--wire_width", type=int, default=10000) # In practice, wire_width shall be set around 1/2 of the memristor size; Hu/MF: 10um; Ferro:200nm;
parser.add_argument("--CMOS_technode", type=int, default=45)
parser.add_argument("--device_roadmap", type=str, default='HP') # HP: High Performance or LP: Low Power
parser.add_argument("--temperature", type=int, default=300)
parser.add_argument("--hardware_estimation", type=int, default=True)
args = parser.parse_args()

def main():
    # seed = args.seed # Fixe Seed
    seed = int(time.time()) # Random Seed
    gpu = args.gpu
    # Sets up Gpu use
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))

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
    _read_batch = args.read_batch
    _batch_size = args.write_batch_size
    _logs = ['test_data', None, False, False, None]

    sim_params = {'device_structure': args.memristor_structure, 'device_name': args.memristor_device,
                  'c2c_variation': args.c2c_variation, 'd2d_variation': args.d2d_variation,
                  'wire_width': args.wire_width, 'input_bit': args.input_bit,
                  'batch_interval': 1, 'CMOS_technode': args.CMOS_technode, 'ADC_precision': args.ADC_precision,
                  'ADC_setting': args.ADC_setting, 'ADC_rounding_function': args.ADC_rounding_function,
                  'device_roadmap': args.device_roadmap, 'temperature': args.temperature,
                  'hardware_estimation': args.hardware_estimation}

    # Run crossbar size experiments
    # size_list = [4,8,16,32,48,64,128,256,512,1024]
    # size_list = [48, 64, 128, 256, 512, 1024]
    size_list = [_rows]
    # size_list = [2048, 256]
    for _rows in size_list:
        _crossbar_1 = MimoMapping(sim_params=sim_params, shape=(_rows, _cols))
        _crossbar_1.to(device)
        _crossbar_2 = MimoMapping(sim_params=sim_params, shape=(_rows, _cols))
        _crossbar_2.to(device)
        _crossbar_3 = MimoMapping(sim_params=sim_params, shape=(_rows, _cols))
        _crossbar_3.to(device)
        _crossbar_4 = MimoMapping(sim_params=sim_params, shape=(_rows, _cols))
        _crossbar_4.to(device)

        # Area print
        if sim_params['hardware_estimation']:
            _crossbar_1.total_area_calculation()
            _crossbar_2.total_area_calculation()     
            _crossbar_3.total_area_calculation()
            _crossbar_4.total_area_calculation()
            print("total area=", _crossbar_1.sim_area['sim_total_area'] + _crossbar_2.sim_area['sim_total_area'] +
                  _crossbar_3.sim_area['sim_total_area'] + _crossbar_4.sim_area['sim_total_area'], " m2")

        run_complex_sim(_crossbar_1, _crossbar_2, _crossbar_3, _crossbar_4, _rep, _read_batch, _batch_size, _rows, _cols, sim_params, device, _logs)


if __name__ == "__main__":
    main()

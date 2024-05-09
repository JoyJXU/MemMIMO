import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')

from testbenches import *
from simbrain.mapping import MimoMapping

#############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rows", type=int, default=8)
parser.add_argument("--cols", type=int, default=1)
parser.add_argument("--rep", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=10000)
parser.add_argument("--memristor_structure", type=str, default='mimo') # trace, mimo or crossbar 
parser.add_argument("--memristor_device", type=str, default='new_ferro') # ideal, ferro, or hu
parser.add_argument("--c2c_variation", type=bool, default=False)
parser.add_argument("--d2d_variation", type=int, default=0) # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
parser.add_argument("--stuck_at_fault", type=bool, default=False)
parser.add_argument("--retention_loss", type=int, default=0) # retention loss, 0: without it, 1: during pulse, 2: no pluse for a long time
parser.add_argument("--aging_effect", type=int, default=0) # 0: No aging effect, 1: equation 1, 2: equation 2
parser.add_argument("--input_bit", type=int, default=8)
parser.add_argument("--ADC_precision", type=int, default=32)
parser.add_argument("--ADC_setting", type=int, default=4)  # 2:two memristor crossbars use one ADC; 4:one memristor crossbar use one ADC
parser.add_argument("--ADC_rounding_function", type=str, default='floor')  # floor or round
parser.add_argument("--wire_width", type=int, default=200) # In practice, wire_width shall be set around 1/2 of the memristor size; Hu: 10um; Ferro:200nm;
parser.add_argument("--CMOS_technode", type=int, default=14)
parser.add_argument("--device_roadmap", type=str, default='HP') # HP: High Performance or LP: Low Power
parser.add_argument("--temperature", type=int, default=300)
parser.add_argument("--hardware_estimation", type=int, default=True)
args = parser.parse_args()

def main():
    # seed = args.seed # Fixe Seed
    seed = int(time.time()) # Random Seed
    gpu = args.gpu
    # Sets up Gpu use
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))

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
    _logs = ['test_data', None, False, False, None]

    sim_params = {'device_structure': args.memristor_structure, 'device_name': args.memristor_device,
                  'c2c_variation': args.c2c_variation, 'd2d_variation': args.d2d_variation,
                  'stuck_at_fault': args.stuck_at_fault, 'retention_loss': args.retention_loss,
                  'aging_effect': args.aging_effect, 'wire_width': args.wire_width, 'input_bit': args.input_bit,
                  'batch_interval': 1, 'CMOS_technode': args.CMOS_technode, 'ADC_precision': args.ADC_precision,
                  'ADC_setting': args.ADC_setting, 'ADC_rounding_function': args.ADC_rounding_function,
                  'device_roadmap': args.device_roadmap, 'temperature': args.temperature,
                  'hardware_estimation': args.hardware_estimation}

    # Run crossbar size experiments
    # size_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    size_list = [16]
    # size_list = [2048, 256]
    for _rows in size_list:
        _crossbar = MimoMapping(sim_params=sim_params, shape=(_rows, _cols))
        _crossbar.to(device)

        # Area print
        if sim_params['hardware_estimation']:
            _crossbar.total_area_calculation()
            print("total area=", _crossbar.sim_area['sim_total_area'], " m2")

        # run_d2d_sim(_crossbar, _rep, _batch_size, _rows, _cols, sim_params, device, _logs)
        run_crossbar_size_sim(_crossbar, _rep, _batch_size, _rows, _cols, sim_params, device, _logs)

    # # plot
    # plt.figure(figsize=(13, 4.5))
    # grid = plt.GridSpec(9, 24, wspace=0.5, hspace=0.5)
    # ax = plt.subplot(grid[0:4, 0:4])
    # bx = plt.subplot(grid[5:9, 0:4])
    # cx = plt.subplot(grid[0:4, 5:9])
    # dx = plt.subplot(grid[5:9, 5:9])
    # ex = plt.subplot(grid[0:4, 10:14])
    # fx = plt.subplot(grid[5:9, 10:14])
    # gx = plt.subplot(grid[0:4, 15:19])
    # hx = plt.subplot(grid[5:9, 15:19])
    # ix = plt.subplot(grid[0:4, 20:24])
    # jx = plt.subplot(grid[5:9, 20:24])
    # figs = [ax, bx, cx, dx, ex, fx, gx, hx, ix, jx]
    #
    # # Run crossbar size experiments
    # # size_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # size_list = [16]
    # # size_list = [2048, 256]
    # for _rows in size_list:
    #     _crossbar = MimoMapping(sim_params=sim_params, shape=(_rows, _cols))
    #     _crossbar.to(device)
    #
    #     # Area print
    #     _crossbar.total_area_calculation()
    #     print("total crossbar area=", _crossbar.sim_area['mem_area'], " m2")
    #
    #     run_crossbar_size_sim(_crossbar, _rep, _batch_size, _rows, _cols, sim_params, device, _logs, figs)


if __name__ == "__main__":
    main()

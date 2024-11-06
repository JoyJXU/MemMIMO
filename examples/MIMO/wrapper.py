import sys
import os

SIMBRAIN_PATH = os.getenv("SIMBRAIN_PATH")

sys.path.append(SIMBRAIN_PATH)

from simbrain.mapping import MimoMapping
from typing import Iterable, Optional
import torch
from itertools import chain

# TODO this values need to be set according to the size of the crossbar and have to be the same as the C FLI interface and the VHDL
CRB_ROW = 3
CRB_COL = 2


class Interface:
    # language=rst
    """
    Interface class for connecting RTL to pytorch-based simulation framework for memristor-based MIMO.
    """

    def __init__(self, shape: Optional[Iterable[int]] = None):
        self.shape = (CRB_ROW, CRB_COL)  # Initialize as 1x1 memristor, in case error
        # self.shape = shape  # (no_rows, no_cols)
        print(" Setting up crossbar, ", CRB_ROW, CRB_COL)
        # parameter setting for memristor crossbar simulation - constants
        # Memristor Array
        memristor_structure = "mimo"
        memristor_device = "ferro"  # ideal, ferro, hu(FS), or MF
        c2c_variation = False
        d2d_variation = 0  # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
        stuck_at_fault = False
        retention_loss = 0  # retention loss, 0: without it, 1: during pulse, 2: no pluse for a long time
        aging_effect = 0  # 0: No aging effect, 1: equation 1, 2: equation 2
        # Peripheral Circuit
        wire_width = 10000  # In practice, wire_width shall be set around 1/2 of the memristor size; Hu: 10um; Ferro:200nm;
        input_bit = 8
        CMOS_technode = 14
        ADC_precision = 16
        ADC_setting = 4  # 2:two memristor crossbars use one ADC; 4:one memristor crossbar use one ADC
        ADC_rounding_function = "floor"  # floor or round
        device_roadmap = "HP"  # HP: High Performance or LP: Low Power
        temperature = 300
        hardware_estimation = True  # area and power estimation

        self.sim_params = {
            "device_structure": memristor_structure,
            "device_name": memristor_device,
            "c2c_variation": c2c_variation,
            "d2d_variation": d2d_variation,
            "wire_width": wire_width,
            "input_bit": input_bit,
            "batch_interval": 1,
            "CMOS_technode": CMOS_technode,
            "ADC_precision": ADC_precision,
            "ADC_setting": ADC_setting,
            "ADC_rounding_function": ADC_rounding_function,
            "device_roadmap": device_roadmap,
            "temperature": temperature,
            "hardware_estimation": hardware_estimation,
        }

        # MimoMaping Initialization
        device = "cpu"
        batch_size = 1
        self.interface = MimoMapping(sim_params=self.sim_params, shape=self.shape)
        self.interface.to(device)
        self.interface.set_batch_size_mimo(batch_size)

    def program(self, matrix):
        # Shape for matrix [write_batch_size, no_row, no_cols]
        self.interface.mapping_write_mimo(target_x=matrix)
        return 1

    def compute(self, vector):
        # Shape for vector [write_batch_size, read_batch_size, no_row]
        output = self.interface.mapping_read_mimo(target_v=vector)
        # Shape for vector [write_batch_size, read_batch_size, no_cols]
        return output


# Instance of Interface
interface_instance = Interface()


def mem_program(value):
    # array to tensor
    matrix_tensor = torch.tensor(value).unsqueeze(0)
    return interface_instance.program(matrix_tensor)


def mem_compute(value):
    # array to tensor
    vector_tensor = torch.tensor(value).unsqueeze(0).unsqueeze(0)
    output_tensor = interface_instance.compute(vector_tensor)
    output_list = output_tensor.tolist()
    output_flatten = list(chain.from_iterable(output_list))
    output_flatten = list(chain.from_iterable(output_flatten))
    # tensor to array
    output = output_flatten  # output_tensor.numpy()
    return output


def main():
    print(SIMBRAIN_PATH)
    x = mem_program([[-0.0574626, 0.000931323], [0.00516884, 0.00190921], [0.018673, 0.145473]])
    print(x)
    x = mem_compute([0.999961, 0.999961, 0.999961])
    print(x)


if __name__ == "__main__":
    main()

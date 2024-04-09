import torch
from typing import Iterable, Optional, Union
import pickle

class Power(torch.nn.Module):
    # language=rst
    """
    Abstract base class for power estimation of memristor crossbar.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
        length_row: float = 0,
        length_col: float = 0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the crossbar.
        :param memristor_info_dict: The parameters of the memristor device.
        :param length_row: The physical length of the horizontal wire in the crossbar.
        :param length_col: The physical length of the vertical wire in the crossbar.
        """
        super().__init__()
    
        self.shape = shape    
        self.device_name = sim_params['device_name']
        self.device_structure = sim_params['device_structure']
        self.average_power = 0
        self.total_energy = 0
        self.read_energy = 0
        self.write_energy = 0
        self.reset_energy = 0
        self.dynamic_read_energy = 0
        self.dynamic_write_energy = 0
        self.dynamic_reset_energy = 0
        self.static_read_energy = 0
        self.static_write_energy = 0
        self.static_reset_energy = 0
        self.register_buffer("selected_write_energy", torch.Tensor())
        self.register_buffer("half_selected_write_energy", torch.Tensor())

        self.v_read = memristor_info_dict[self.device_name]['v_read']
        
        self.wire_cap_row = length_row * 0.2e-15/1e-6
        self.wire_cap_col = length_col * 0.2e-15/1e-6
        self.dt = memristor_info_dict[self.device_name]['delta_t']
        self.dr = memristor_info_dict[self.device_name]['duty_ratio']
    
        self.sim_power = {}

        with open('../../memristor_lut.pkl', 'rb') as f:
            self.memristor_luts = pickle.load(f)
        assert self.device_name in self.memristor_luts.keys(), "No Look-Up-Table Data Available for the Target Memristor Type!"
        

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.selected_write_energy = torch.zeros(batch_size, *self.shape, device=self.selected_write_energy.device)
        self.half_selected_write_energy = torch.zeros(batch_size, *self.shape, device=self.half_selected_write_energy.device)


    def read_energy_calculation(self, mem_v_bool, mem_c, total_wire_resistance) -> None:
        # language=rst
        """
        Calculate read energy for memrisotr crossbar. Called when the crossbar is read.

        :param mem_v_bool: Read voltage, shape [batchsize, read_no=1, crossbar_row], type: bool.
        :param mem_c: Memristor crossbar conductance, shape [batchsize, crossbar_row, crossbar_col].
        :param total_wire_resistance: Wire resistance for every memristor in the crossbar, shape [batchsize, crossbar_row, crossbar_col].
        """
        # Use nonzero instead of torch.sum() to save memory
        v_sum = torch.nonzero(mem_v_bool).size(0)
        self.dynamic_read_energy += self.v_read ** 2 * v_sum * self.wire_cap_row

        mem_r = 1.0 / mem_c
        mem_r = mem_r + total_wire_resistance.unsqueeze(0)
        memristor_c = 1.0 / mem_r
        for i in range(mem_v_bool.size(0)):
            self.static_read_energy += self.v_read ** 2 * torch.sum(torch.matmul(mem_v_bool[i].float(), memristor_c)) * self.dt * self.dr

        self.read_energy = self.dynamic_read_energy + self.static_read_energy


    def write_energy_calculation(self, mem_v, mem_c, mem_c_pre, total_wire_resistance) -> None:
        # language=rst
        """
        Calculate write energy for memrisotr crossbar. Called when the crossbar is wrote.

        :param mem_v: Write voltage, shape [batchsize, crossbar_row, crossbar_col].
        :param mem_c: Memristor crossbar conductance after write, shape [batchsize, crossbar_row, crossbar_col].
        :param mem_c_pre: Memristor crossbar conductance before write, shape [batchsize, crossbar_row, crossbar_col].
        :param total_wire_resistance: Wire resistance for every memristor in the crossbar, shape [batchsize, crossbar_row, crossbar_col].
        """
        if self.device_structure == 'trace':
            # Dynamic write energy
            self.dynamic_write_energy += torch.sum(mem_v * mem_v * self.wire_cap_col)

            # Static write energy
            mem_r = 1.0 / (1 / 2 * (mem_c + mem_c_pre))
            mem_r = mem_r + total_wire_resistance.unsqueeze(0)
            mem_c = 1.0 / mem_r
            self.selected_write_energy = mem_v * mem_v * self.dr * self.dt * mem_c
            self.static_write_energy += torch.sum(self.selected_write_energy)

        elif self.device_structure in {'crossbar', 'mimo'}:
            V_write = self.memristor_luts[self.device_name]['voltage']

            # Col cap dynamic write energy
            self.dynamic_write_energy += torch.sum((V_write - mem_v) * (V_write - mem_v) * self.wire_cap_col)

            # Row cap dynamic write energy
            # 1 selected using V_write; (self.shape[0] - 1) half selected using 1/2 V_write
            self.dynamic_write_energy += self.shape[0] * V_write * V_write * self.wire_cap_row * (1 + 1 / 4 * (
                        self.shape[0] - 1))

            # static write energy
            # Seleceted write energy
            selected_mem_r = 1.0 / (1 / 2 * (mem_c + mem_c_pre))
            selected_mem_r = selected_mem_r + total_wire_resistance.unsqueeze(0)
            selected_mem_c = 1.0 / selected_mem_r
            self.selected_write_energy = (mem_v * mem_v * self.dr * self.dt * selected_mem_c)

            # half selected write energy
            r_after = 1.0 / mem_c
            r_after = r_after + total_wire_resistance.unsqueeze(0)
            c_after = 1.0 / r_after

            r_pre = 1.0 / mem_c_pre
            r_pre = r_pre + total_wire_resistance.unsqueeze(0)
            c_pre = 1.0 / r_pre

            counter = torch.arange(self.shape[0], device=self.selected_write_energy.device)
            self.half_selected_write_energy = (1 / 2 * V_write) * (1 / 2 * V_write) * self.dr * self.dt * (
                        counter[None, :, None] * c_pre + counter.flip(0)[None, :, None] * c_after)

            self.static_write_energy += torch.sum(self.selected_write_energy) + torch.sum(self.half_selected_write_energy)

            # # static write energy
            # for write_row in range(self.shape[0]):
                # # Selected write energy
                # selected_mem_r = 1.0 / (1/2 * (mem_c[:, write_row, :] + mem_c_pre[:, write_row, :]))
                # selected_mem_r = selected_mem_r + total_wire_resistance[write_row, :].unsqueeze(0)
                # selected_mem_c = 1.0 / selected_mem_r
                # self.selected_write_energy = (mem_v[:, write_row, :] * mem_v[:, write_row, :] * 1/2 * self.dt * selected_mem_c).unsqueeze(1)

                # # half selected write energy
                # self.half_selected_write_energy.zero_()
                #
                # half_selected_mem_r = 1.0 / mem_c[:, 0:write_row, :]
                # half_selected_mem_r = half_selected_mem_r + total_wire_resistance[0:write_row, :].unsqueeze(0)
                # half_selected_mem_c = 1.0 / half_selected_mem_r
                # self.half_selected_write_energy[:, 0:write_row, :] = (1/2 * V_write) * (1/2 * V_write) * 1/2 * self.dt * half_selected_mem_c
                #
                # half_selected_mem_r = 1.0 / mem_c_pre[:, write_row+1:, :]
                # half_selected_mem_r = half_selected_mem_r + total_wire_resistance[write_row+1:, :].unsqueeze(0)
                # half_selected_mem_c = 1.0 / half_selected_mem_r
                # self.half_selected_write_energy[:, write_row+1:, :] = (1 / 2 * V_write) * (1 / 2 * V_write) * 1/2 * self.dt * half_selected_mem_c
                #
                # self.static_write_energy += torch.sum(self.selected_write_energy) + torch.sum(self.half_selected_write_energy)

        else:
            raise Exception("Only trace, mimo and crossbar architecture are supported!")

        self.write_energy = self.dynamic_write_energy + self.static_write_energy


    def reset_energy_calculation(self, mem_v, mem_c, mem_c_pre, total_wire_resistance) -> None:
        # language=rst
        """
        Calculate reset energy for memrisotr crossbar. Called when the crossbar is reset.

        :param mem_v: Reset voltage, shape [batchsize, crossbar_row, crossbar_col].
        :param mem_c: Memristor crossbar conductance after reset, shape [batchsize, crossbar_row, crossbar_col].
        :param mem_c_pre: Memristor crossbar conductance before reset, shape [batchsize, crossbar_row, crossbar_col].
        :param total_wire_resistance: Wire resistance for every memristor in the crossbar, shape [batchsize, crossbar_row, crossbar_col].
        """
        if self.device_structure == 'trace':
            raise Exception("In the trace architecture, mem_write is used instead of mem_reset!")

        elif self.device_structure in {'crossbar', 'mimo'}:
            # Dynamic reset energy
            self.dynamic_reset_energy += torch.sum(mem_v[:, 0, :] * mem_v[:, 0, :] * self.wire_cap_col)

            # Static write energy
            mem_r = 1.0 / (1 / 2 * (mem_c + mem_c_pre))
            mem_r = mem_r + total_wire_resistance.unsqueeze(0)
            mem_c = 1.0 / mem_r
            self.static_reset_energy += torch.sum(mem_v * mem_v * self.dr * self.dt * mem_c) #?why 1/2

        else:
            raise Exception("Only trace, mimo and crossbar architecture are supported!")

        self.reset_energy = self.dynamic_reset_energy + self.static_reset_energy


    def total_energy_calculation(self, mem_t) -> None:
        # language=rst
        """
        Calculate total energy for memrisotr crossbar. Called when power is reported.

        :param mem_t: Time of the memristor crossbar.
        """
        self.total_energy = self.read_energy + self.write_energy + self.reset_energy
        self.average_power = self.total_energy / (torch.max(mem_t) * self.dt)
        self.sim_power = {'dynamic_read_energy': self.dynamic_read_energy,
                          'dynamic_write_energy': self.dynamic_write_energy,
                          'dynamic_reset_energy': self.dynamic_reset_energy,
                          'static_read_energy': self.static_read_energy,
                          'static_write_energy': self.static_write_energy,
                          'static_reset_energy': self.static_reset_energy,
                          'read_energy': self.read_energy,
                          'write_energy': self.write_energy,
                          'reset_energy': self.reset_energy,
                          'total_energy': self.total_energy,
                          'time': torch.max(mem_t) * self.dt,
                          'average_power': self.average_power}
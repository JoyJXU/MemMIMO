from typing import Iterable, Optional, Union
from simbrain.mempower import Power
from simbrain.memarea import Area
import torch
import json
import os

SIMBRAIN_PATH = os.getenv("SIMBRAIN_PATH")

class MemristorArray(torch.nn.Module):
    # language=rst
    """
    Abstract base class for memristor arrays.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the crossbar.
        :param memristor_info_dict: The parameters of the memristor device.
        """
        super().__init__()
    
        self.shape = shape    
    
        self.register_buffer("mem_x", torch.Tensor())  # Memristor-based firing traces.
        self.register_buffer("mem_c", torch.Tensor())
        self.register_buffer("mem_c_pre", torch.Tensor())
        self.register_buffer("mem_i", torch.Tensor())
        self.register_buffer("mem_t", torch.Tensor())

        self.device_structure = sim_params['device_structure']
        self.device_name = sim_params['device_name']
        self.c2c_variation = sim_params['c2c_variation']
        self.d2d_variation = sim_params['d2d_variation']

        if self.c2c_variation:
            self.register_buffer("normal_absolute", torch.Tensor())
            self.register_buffer("normal_relative", torch.Tensor())
    
        if self.d2d_variation in [1, 2]:
            self.register_buffer("Gon_d2d", torch.Tensor())
            self.register_buffer("Goff_d2d", torch.Tensor())
            
        if self.d2d_variation in [1, 3]:
            self.register_buffer("Pon_d2d", torch.Tensor())
            self.register_buffer("Poff_d2d", torch.Tensor())

        self.memristor_info_dict = memristor_info_dict
        self.dt = self.memristor_info_dict[self.device_name]['delta_t']
        self.dr = self.memristor_info_dict[self.device_name]['duty_ratio']
        self.batch_size = None

        file_path = os.path.join(SIMBRAIN_PATH, 'wire_tech_info.json')
        with open(file_path, 'r') as file:
            self.tech_info_dict = json.load(file)

        self.input_bit = sim_params['input_bit']

        self.wire_width = sim_params['wire_width']
        relax_ratio_col = self.memristor_info_dict[self.device_name]['relax_ratio_col'] # Leave space for adjacent memristors
        relax_ratio_row = self.memristor_info_dict[self.device_name]['relax_ratio_row'] # Leave space for adjacent memristors
        mem_size = self.memristor_info_dict[self.device_name]['mem_size'] * 1e-9
        self.length_row = shape[1] * relax_ratio_col * mem_size
        self.length_col = shape[0] * relax_ratio_row * mem_size
        AR = self.tech_info_dict[str(self.wire_width)]['AR']
        Rho = self.tech_info_dict[str(self.wire_width)]['Rho']
        wire_resistance_unit_col = relax_ratio_col * mem_size * Rho / (AR * self.wire_width * self.wire_width * 1e-18)
        wire_resistance_unit_row = relax_ratio_row * mem_size * Rho / (AR * self.wire_width * self.wire_width * 1e-18)
        self.register_buffer("total_wire_resistance", torch.Tensor())
        self.total_wire_resistance = wire_resistance_unit_col * torch.arange(1, self.shape[1] + 1, device=self.total_wire_resistance.device) + \
            wire_resistance_unit_row * torch.arange(self.shape[0], 0, -1, device=self.total_wire_resistance.device)[:, None]

        self.hardware_estimation = sim_params['hardware_estimation']
        if self.hardware_estimation:
            self.power = Power(sim_params=sim_params, shape=self.shape, memristor_info_dict=self.memristor_info_dict, length_row=self.length_row, length_col=self.length_col)
            self.area = Area(sim_params=sim_params, shape=self.shape, memristor_info_dict=self.memristor_info_dict, length_row=self.length_row, length_col=self.length_col)


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size

        self.mem_x = torch.zeros(batch_size, *self.shape, device=self.mem_x.device)
        self.mem_c = torch.ones(batch_size, *self.shape, device=self.mem_c.device) * \
                     self.memristor_info_dict[self.device_name]['G_on']
        self.mem_c_pre = torch.ones(batch_size, *self.shape, device=self.mem_c_pre.device) * \
                     self.memristor_info_dict[self.device_name]['G_on']
        self.mem_t = torch.zeros(batch_size, *self.shape, device=self.mem_t.device)
        self.mem_i = torch.zeros(batch_size, 1, self.shape[1], device=self.mem_i.device)

        if self.c2c_variation:
            self.normal_relative = torch.zeros(batch_size, *self.shape, device=self.normal_relative.device)
            self.normal_absolute = torch.zeros(batch_size, *self.shape, device=self.normal_absolute.device)

        if self.d2d_variation in [1, 2]:
            # print('Add D2D variation in Gon/Goff!')
            G_off = self.memristor_info_dict[self.device_name]['G_off']
            G_on = self.memristor_info_dict[self.device_name]['G_on']
            Gon_sigma = self.memristor_info_dict[self.device_name]['Gon_sigma']
            Goff_sigma = self.memristor_info_dict[self.device_name]['Goff_sigma']

            # Initialize
            self.Gon_d2d = torch.zeros(*self.shape, device=self.Gon_d2d.device)
            self.Goff_d2d = torch.zeros(*self.shape, device=self.Goff_d2d.device)
            # Add d2d variation
            self.Gon_d2d.normal_(mean=G_on, std=Gon_sigma * G_on)
            self.Goff_d2d.normal_(mean=G_off, std=Goff_sigma * G_off)
            # Clipping
            self.Gon_d2d = torch.clamp(self.Gon_d2d, min=0)
            self.Goff_d2d = torch.clamp(self.Goff_d2d, min=0)

            self.Gon_d2d = torch.stack([self.Gon_d2d] * batch_size)
            self.Goff_d2d = torch.stack([self.Goff_d2d] * batch_size)

        if self.d2d_variation in [1, 3]:
            # print('Add D2D variation in Pon/Poff!')
            P_off = self.memristor_info_dict[self.device_name]['P_off']
            P_on = self.memristor_info_dict[self.device_name]['P_on']
            Pon_sigma = self.memristor_info_dict[self.device_name]['Pon_sigma']
            Poff_sigma = self.memristor_info_dict[self.device_name]['Poff_sigma']

            # Initialize
            self.Pon_d2d = torch.zeros(*self.shape, device=self.Pon_d2d.device)
            self.Poff_d2d = torch.zeros(*self.shape, device=self.Poff_d2d.device)
            # Add d2d variation
            self.Pon_d2d.normal_(mean=P_on, std=Pon_sigma * P_on)
            self.Poff_d2d.normal_(mean=P_off, std=Poff_sigma * P_off)
            # Clipping
            self.Pon_d2d = torch.clamp(self.Pon_d2d, min=0)
            self.Poff_d2d = torch.clamp(self.Poff_d2d, min=0)

            self.Pon_d2d = torch.stack([self.Pon_d2d] * batch_size)
            self.Poff_d2d = torch.stack([self.Poff_d2d] * batch_size)

        if self.hardware_estimation:
            self.power.set_batch_size(batch_size=self.batch_size)


    def memristor_write(self, mem_v: torch.Tensor):
        # language=rst
        """
        Memristor write operation for a single simulation step.
    
        :param mem_v: Voltage inputs to the memristor array.
        """

        mem_info = self.memristor_info_dict[self.device_name]
        k_off = mem_info['k_off']
        k_on = mem_info['k_on']
        v_off = mem_info['v_off']
        v_on = mem_info['v_on']
        alpha_off = mem_info['alpha_off']
        alpha_on = mem_info['alpha_on']
        P_off = mem_info['P_off']
        P_on = mem_info['P_on']
        G_off = mem_info['G_off']
        G_on = mem_info['G_on']
        sigma_relative = mem_info['sigma_relative']
        sigma_absolute = mem_info['sigma_absolute']

        self.mem_t += self.shape[0]
        self.mem_c_pre = self.mem_c.clone()

        if self.d2d_variation in [1, 3]:
            self.mem_x = torch.where(mem_v >= v_off, \
                                     self.mem_x + self.dt * self.dr * (k_off * (mem_v / v_off - 1) ** alpha_off) * ( \
                                     1 - self.mem_x) ** self.Poff_d2d, self.mem_x)
                        
            self.mem_x = torch.where(mem_v <= v_on, \
                                     self.mem_x + self.dt * self.dr * (k_on * (mem_v / v_on - 1) ** alpha_on) * ( \
                                     self.mem_x) ** self.Pon_d2d, self.mem_x)
    
        else:
            self.mem_x = torch.where(mem_v >= v_off, \
                                    self.mem_x + self.dt * self.dr * (k_off * (mem_v / v_off - 1) ** alpha_off) * ( \
                                    1 - self.mem_x) ** P_off, self.mem_x)

            self.mem_x = torch.where(mem_v <= v_on, \
                                    self.mem_x + self.dt * self.dr * (k_on * (mem_v / v_on - 1) ** alpha_on) * ( \
                                    self.mem_x) ** P_on, self.mem_x)
    
        self.mem_x = torch.clamp(self.mem_x, min=0, max=1)

        if self.c2c_variation:
            self.normal_relative.normal_(mean=0., std=sigma_relative)
            self.normal_absolute.normal_(mean=0., std=sigma_absolute)
    
            device_v = torch.mul(self.mem_x, self.normal_relative) + self.normal_absolute
            self.x2 = self.mem_x + device_v
    
            self.x2 = torch.clamp(self.x2, min=0, max=1)
    
        else:
            self.x2 = self.mem_x
    
        if self.d2d_variation in [1, 2]:
            self.mem_c = self.Goff_d2d * self.x2 + self.Gon_d2d * (1 - self.x2)
        else:
            self.mem_c = G_off * self.x2 + G_on * (1 - self.x2)
        
        if self.hardware_estimation:
            self.power.write_energy_calculation(mem_v=mem_v, mem_c=self.mem_c, mem_c_pre=self.mem_c_pre, total_wire_resistance=self.total_wire_resistance)
        
        return self.mem_c


    def memristor_read(self, mem_v: torch.Tensor):
        # language=rst
        """
        Memristor read operation for a single simulation step.

        :param mem_v: Voltage inputs to the memristor array, type: bool
        """
        # Detect v_read and threshold voltage
        mem_info = self.memristor_info_dict[self.device_name]
        v_read = mem_info['v_read']
        v_off = mem_info['v_off']
        v_on = mem_info['v_on']
        in_threshold = (v_read >= v_on) & (v_read <= v_off)
        assert in_threshold, "Read Voltage of the Memristor Array Exceeds the Threshold Voltage!"

        # Take the wire resistance into account
        mem_r = 1.0 / self.mem_c
        mem_r = mem_r + self.total_wire_resistance.unsqueeze(0)
        mem_c = 1.0 / mem_r

        # vector multiplication:
        # mem_v shape: [input_bit, batchsize, read_no=1, array_row],
        # mem_array shape: [batchsize, array_row, array_column],
        # output_i shape: [input_bit, batchsize, read_no=1, array_column]
        self.mem_i = torch.matmul(mem_v * v_read, mem_c)

        # Non-idealities
        mem_info = self.memristor_info_dict[self.device_name]
        v_off = mem_info['v_off']
        v_on = mem_info['v_on']
        G_off = mem_info['G_off']
        G_on = mem_info['G_on']

        if self.c2c_variation:
            device_v = torch.mul(self.mem_x, self.normal_relative) + self.normal_absolute
            self.x2 = self.mem_x + device_v
            self.x2 = torch.clamp(self.x2, min=0, max=1)
        else:
            self.x2 = self.mem_x

        if self.d2d_variation in [1, 2]:
            self.mem_c = self.Goff_d2d * self.x2 + self.Gon_d2d * (1 - self.x2)
        else:
            self.mem_c = G_off * self.x2 + G_on * (1 - self.x2)

        # mem_t update according to the sequential read
        self.mem_t += mem_v.shape[0] * mem_v.shape[2]

        if self.hardware_estimation:
            self.power.read_energy_calculation(mem_v_bool=mem_v, mem_c=self.mem_c, total_wire_resistance=self.total_wire_resistance)

        return self.mem_i


    def memristor_reset(self, mem_v: torch.Tensor):
        # language=rst
        """
        Memristor reset operation for a single simulation step.

        :param mem_v: Voltage inputs to the memristor array.
        """

        mem_info = self.memristor_info_dict[self.device_name]
        k_off = mem_info['k_off']
        k_on = mem_info['k_on']
        v_off = mem_info['v_off']
        v_on = mem_info['v_on']
        alpha_off = mem_info['alpha_off']
        alpha_on = mem_info['alpha_on']
        P_off = mem_info['P_off']
        P_on = mem_info['P_on']
        G_off = mem_info['G_off']
        G_on = mem_info['G_on']
        sigma_relative = mem_info['sigma_relative']
        sigma_absolute = mem_info['sigma_absolute']

        self.mem_t += 1
        self.mem_c_pre = self.mem_c.clone()

        if self.d2d_variation in [1, 3]:
            self.mem_x = torch.where(mem_v >= v_off, \
                                     self.mem_x + self.dt * self.dr * (k_off * (mem_v / v_off - 1) ** alpha_off) * ( \
                                                 1 - self.mem_x) ** self.Poff_d2d, self.mem_x)

            self.mem_x = torch.where(mem_v <= v_on, \
                                     self.mem_x + self.dt * self.dr * (k_on * (mem_v / v_on - 1) ** alpha_on) * ( \
                                         self.mem_x) ** self.Pon_d2d, self.mem_x)

        else:
            self.mem_x = torch.where(mem_v >= v_off, \
                                     self.mem_x + self.dt * self.dr * (k_off * (mem_v / v_off - 1) ** alpha_off) * ( \
                                                 1 - self.mem_x) ** P_off, self.mem_x)

            self.mem_x = torch.where(mem_v <= v_on, \
                                     self.mem_x + self.dt * self.dr * (k_on * (mem_v / v_on - 1) ** alpha_on) * ( \
                                         self.mem_x) ** P_on, self.mem_x)

        self.mem_x = torch.clamp(self.mem_x, min=0, max=1)

        if self.c2c_variation:
            self.normal_relative.normal_(mean=0., std=sigma_relative)
            self.normal_absolute.normal_(mean=0., std=sigma_absolute)

            device_v = torch.mul(self.mem_x, self.normal_relative) + self.normal_absolute
            self.x2 = self.mem_x + device_v

            self.x2 = torch.clamp(self.x2, min=0, max=1)

        else:
            self.x2 = self.mem_x

        if self.d2d_variation in [1, 2]:
            self.mem_c = self.Goff_d2d * self.x2 + self.Gon_d2d * (1 - self.x2)
        else:
            self.mem_c = G_off * self.x2 + G_on * (1 - self.x2)

        if self.hardware_estimation:
            self.power.reset_energy_calculation(mem_v=mem_v, mem_c=self.mem_c, mem_c_pre=self.mem_c_pre,
                                            total_wire_resistance=self.total_wire_resistance)

        return self.mem_c


    def total_energy_calculation(self) -> None:
        self.power.total_energy_calculation(mem_t=self.mem_t)

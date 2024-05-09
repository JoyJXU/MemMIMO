from typing import Iterable, Optional, Union
from simbrain.mempower import Power
from simbrain.memarea import Area
import torch
import json

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
        self.stuck_at_fault = sim_params['stuck_at_fault']
        self.retention_loss = sim_params['retention_loss']
        self.aging_effect = sim_params['aging_effect']
    
        if self.c2c_variation:
            self.register_buffer("normal_absolute", torch.Tensor())
            self.register_buffer("normal_relative", torch.Tensor())
    
        if self.d2d_variation in [1, 2]:
            self.register_buffer("Gon_d2d", torch.Tensor())
            self.register_buffer("Goff_d2d", torch.Tensor())
            
        if self.d2d_variation in [1, 3]:
            self.register_buffer("Pon_d2d", torch.Tensor())
            self.register_buffer("Poff_d2d", torch.Tensor())
    
        if self.stuck_at_fault:
            self.register_buffer("SAF0_mask", torch.Tensor())
            self.register_buffer("SAF1_mask", torch.Tensor())
            
        if self.retention_loss == 2:
            self.register_buffer("mem_v_threshold", torch.Tensor())
            self.register_buffer("mem_loss_time", torch.Tensor())  
        
        if self.aging_effect:
            self.register_buffer("Gon_aging", torch.Tensor())
            self.register_buffer("Goff_aging", torch.Tensor())
            self.register_buffer("Gon_0", torch.Tensor())
            self.register_buffer("Goff_0", torch.Tensor())

        if self.stuck_at_fault:
            self.register_buffer("SAF0_mask", torch.Tensor())
            self.register_buffer("SAF1_mask", torch.Tensor())
            self.register_buffer("Q_mask", torch.Tensor())

        self.memristor_info_dict = memristor_info_dict
        self.dt = self.memristor_info_dict[self.device_name]['delta_t']
        self.dr = self.memristor_info_dict[self.device_name]['duty_ratio']
        self.batch_size = None

        with open('../../wire_tech_info.json', 'r') as file:
            self.tech_info_dict = json.load(file)

        self.input_bit = sim_params['input_bit']

        self.wire_width = sim_params['wire_width']
        relax_ratio_col = self.memristor_info_dict[self.device_name]['relax_ratio_col'] # Leave space for adjacent memristors
        relax_ratio_row = self.memristor_info_dict[self.device_name]['relax_ratio_col'] # Leave space for adjacent memristors
        mem_size = self.memristor_info_dict[self.device_name]['mem_size'] * 1e-9
        self.length_row = shape[1] * relax_ratio_col * mem_size
        self.length_col = shape[0] * relax_ratio_row * mem_size
        AR = self.tech_info_dict[str(self.wire_width)]['AR']
        Rho = self.tech_info_dict[str(self.wire_width)]['Rho']
        wire_resistance_unit_col = relax_ratio_col * mem_size * Rho / (AR * self.wire_width * self.wire_width * 1e-18)
        wire_resistance_unit_row = relax_ratio_col * mem_size * Rho / (AR * self.wire_width * self.wire_width * 1e-18)
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
            print('Add D2D variation in Gon/Goff!')
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
            print('Add D2D variation in Pon/Poff!')
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

        if self.aging_effect:
            # Initialize the time-dependent Gon/Goff
            self.Gon_aging = torch.zeros(*self.shape, device=self.Gon_aging.device)
            self.Goff_aging = torch.zeros(*self.shape, device=self.Goff_aging.device)
            self.Gon_aging = torch.stack([self.Goff_aging] * self.batch_size)
            self.Goff_aging = torch.stack([self.Goff_aging] * self.batch_size)

            # Initialize the start point Gon/Goff
            if self.d2d_variation in [1, 2]:
                self.Gon_0 = self.Gon_d2d
                self.Goff_0 = self.Goff_d2d
            else:
                self.Gon_0 = self.memristor_info_dict[self.device_name]['G_on'] * torch.ones(batch_size, *self.shape, device=self.Gon_0.device)
                self.Goff_0 = self.memristor_info_dict[self.device_name]['G_off'] * torch.ones(batch_size, *self.shape, device=self.Goff_0.device)

        if self.stuck_at_fault:
            SAF_lambda = self.memristor_info_dict[self.device_name]['SAF_lambda']
            SAF_ratio = self.memristor_info_dict[self.device_name]['SAF_ratio'] # SAF0:SAF1

            # Add pre-deployment SAF #TODO:Change to architectural SAF with Poisson-distributed intra-crossbar and uniform-distributed inter-crossbar SAF
            self.Q_mask = torch.zeros(*self.shape, device=self.Q_mask.device)
            self.SAF0_mask = torch.zeros(*self.shape, device=self.SAF0_mask.device)
            self.SAF1_mask = torch.zeros(*self.shape, device=self.SAF1_mask.device)

            self.Q_mask.uniform_()
            self.SAF0_mask = self.Q_mask < ((SAF_ratio / (SAF_ratio + 1)) * SAF_lambda)
            self.SAF1_mask = (self.Q_mask >= ((SAF_ratio / (SAF_ratio + 1)) * SAF_lambda)) & (self.Q_mask < SAF_lambda)

        if self.retention_loss == 2:
            self.mem_v_threshold = torch.zeros(batch_size, *self.shape, device=self.mem_v_threshold.device)
            self.mem_loss_time = torch.zeros(batch_size, *self.shape, device=self.mem_loss_time.device)

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
        retention_loss_tau = mem_info['retention_loss_tau']
        retention_loss_beta = mem_info['retention_loss_beta']
        Aging_on = mem_info['Aging_on']
        Aging_off = mem_info['Aging_off']

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
    
        # Retention Loss
        if self.retention_loss == 1:
            # G(t) = G(0) * e^(- t*tau)^beta                      
            self.mem_x = G_off * self.mem_x + G_on * (1 - self.mem_x)
            self.mem_x *= torch.exp(torch.tensor(-((1 - self.dr) * self.dt * retention_loss_tau) ** retention_loss_beta))
            self.mem_x = torch.clamp(self.mem_x, min=G_on, max=G_off)
            self.mem_x = (self.mem_x - G_on) / (G_off - G_on)
            # tau = 0.012478 , beta = 1.066  or  tau = 0.01245 , beta = 1.073
            
        if self.retention_loss == 2:
            # dG(t)/dt = - tau^beta * beta * G(t) * t ^ (beta - 1)                    
            self.mem_v_threshold = torch.where((mem_v > v_on) & (mem_v < v_off), torch.zeros_like(mem_v), torch.ones_like(mem_v))
            self.mem_loss_time[self.mem_v_threshold == 0] += self.dt
            self.mem_loss_time[self.mem_v_threshold == 1] = 0         
            self.mem_x = G_off * self.mem_x + G_on * (1 - self.mem_x)
            self.mem_x -= self.mem_x * self.dt * retention_loss_tau ** retention_loss_beta * retention_loss_beta * self.mem_loss_time ** (retention_loss_beta - 1)
            self.mem_x = torch.clamp(self.mem_x, min=G_on, max=G_off)
            self.mem_x = (self.mem_x - G_on) / (G_off - G_on)
    
        if self.c2c_variation:
            self.normal_relative.normal_(mean=0., std=sigma_relative)
            self.normal_absolute.normal_(mean=0., std=sigma_absolute)
    
            device_v = torch.mul(self.mem_x, self.normal_relative) + self.normal_absolute
            self.x2 = self.mem_x + device_v
    
            self.x2 = torch.clamp(self.x2, min=0, max=1)
    
        else:
            self.x2 = self.mem_x
    
        if self.stuck_at_fault:
            self.x2.masked_fill_(self.SAF0_mask, 0)
            self.x2.masked_fill_(self.SAF1_mask, 1)

        if self.aging_effect:
            self.cal_Gon_Goff(Aging_on, Aging_off)
            self.mem_c = self.Goff_aging * self.x2 + self.Gon_aging * (1 - self.x2)

        elif self.d2d_variation in [1, 2]:
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
        retention_loss_tau = mem_info['retention_loss_tau']
        retention_loss_beta = mem_info['retention_loss_beta']
        Aging_on = mem_info['Aging_on']
        Aging_off = mem_info['Aging_off']

        if self.retention_loss == 1:
            # G(t) = G(0) * e^(- t*tau)^beta
            self.mem_x = G_off * self.mem_x + G_on * (1 - self.mem_x)
            self.mem_x *= torch.exp(
                torch.tensor(-((1 - self.dr) * self.dt * retention_loss_tau) ** retention_loss_beta))
            self.mem_x = torch.clamp(self.mem_x, min=G_on, max=G_off)
            self.mem_x = (self.mem_x - G_on) / (G_off - G_on)
            # tau = 0.012478 , beta = 1.066  or  tau = 0.01245 , beta = 1.073

        if self.retention_loss == 2:
            # dG(t)/dt = - tau^beta * beta * G(t) * t ^ (beta - 1)
            self.mem_v_threshold = torch.where((mem_v > v_on) & (mem_v < v_off), torch.zeros_like(mem_v),
                                               torch.ones_like(mem_v))
            self.mem_loss_time[self.mem_v_threshold == 0] += self.dt
            self.mem_loss_time[self.mem_v_threshold == 1] = 0
            self.mem_x = G_off * self.mem_x + G_on * (1 - self.mem_x)
            self.mem_x -= self.mem_x * self.dt * retention_loss_tau ** retention_loss_beta * retention_loss_beta * self.mem_loss_time ** (
                        retention_loss_beta - 1)
            self.mem_x = torch.clamp(self.mem_x, min=G_on, max=G_off)
            self.mem_x = (self.mem_x - G_on) / (G_off - G_on)

        self.x2 = self.mem_x

        if self.stuck_at_fault:
            self.x2.masked_fill_(self.SAF0_mask, 0)
            self.x2.masked_fill_(self.SAF1_mask, 1)

        if self.aging_effect:
            self.cal_Gon_Goff(Aging_on, Aging_off)
            self.mem_c = self.Goff_aging * self.x2 + self.Gon_aging * (1 - self.x2)

        elif self.d2d_variation in [1, 2]:
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
        retention_loss_tau = mem_info['retention_loss_tau']
        retention_loss_beta = mem_info['retention_loss_beta']
        Aging_on = mem_info['Aging_on']
        Aging_off = mem_info['Aging_off']

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

        # Retention Loss
        if self.retention_loss == 1:
            # G(t) = G(0) * e^(- t*tau)^beta
            self.mem_x = G_off * self.mem_x + G_on * (1 - self.mem_x)
            self.mem_x *= torch.exp(torch.tensor(-((1 - self.dr) * self.dt * retention_loss_tau) ** retention_loss_beta))
            self.mem_x = torch.clamp(self.mem_x, min=G_on, max=G_off)
            self.mem_x = (self.mem_x - G_on) / (G_off - G_on)
            # tau = 0.012478 , beta = 1.066  or  tau = 0.01245 , beta = 1.073

        if self.retention_loss == 2:
            # dG(t)/dt = - tau^beta * beta * G(t) * t ^ (beta - 1)
            self.mem_v_threshold = torch.where((mem_v > v_on) & (mem_v < v_off), torch.zeros_like(mem_v),
                                               torch.ones_like(mem_v))
            self.mem_loss_time[self.mem_v_threshold == 0] += self.dt
            self.mem_loss_time[self.mem_v_threshold == 1] = 0
            self.mem_x = G_off * self.mem_x + G_on * (1 - self.mem_x)
            self.mem_x -= self.mem_x * self.dt * retention_loss_tau ** retention_loss_beta * retention_loss_beta * self.mem_loss_time ** (
                        retention_loss_beta - 1)
            self.mem_x = torch.clamp(self.mem_x, min=G_on, max=G_off)
            self.mem_x = (self.mem_x - G_on) / (G_off - G_on)

        if self.c2c_variation:
            self.normal_relative.normal_(mean=0., std=sigma_relative)
            self.normal_absolute.normal_(mean=0., std=sigma_absolute)

            device_v = torch.mul(self.mem_x, self.normal_relative) + self.normal_absolute
            self.x2 = self.mem_x + device_v

            self.x2 = torch.clamp(self.x2, min=0, max=1)

        else:
            self.x2 = self.mem_x

        if self.stuck_at_fault:
            self.x2.masked_fill_(self.SAF0_mask, 0)
            self.x2.masked_fill_(self.SAF1_mask, 1)

        if self.aging_effect:
            self.cal_Gon_Goff(Aging_on, Aging_off)
            self.mem_c = self.Goff_aging * self.x2 + self.Gon_aging * (1 - self.x2)

        elif self.d2d_variation in [1, 2]:
            self.mem_c = self.Goff_d2d * self.x2 + self.Gon_d2d * (1 - self.x2)

        else:
            self.mem_c = G_off * self.x2 + G_on * (1 - self.x2)

        if self.hardware_estimation:
            self.power.reset_energy_calculation(mem_v=mem_v, mem_c=self.mem_c, mem_c_pre=self.mem_c_pre,
                                            total_wire_resistance=self.total_wire_resistance)

        return self.mem_c


    def cal_Gon_Goff(self, age_on, age_off) -> None:
        if self.aging_effect == 1: #equation 1: G=G_0*(1-age)**t
            self.Gon_aging = self.Gon_0 * ((1 - age_on) ** (self.mem_t * self.dt))
            self.Goff_aging = self.Goff_0 * ((1 - age_off) ** (self.mem_t * self.dt))
        elif self.aging_effect == 2: #equation 2: G=age*t+G_0
            self.Gon_aging = age_on * self.mem_t * self.dt + self.Gon_0
            self.Goff_aging = age_off * self.mem_t * self.dt + self.Goff_0


    def update_SAF_mask(self) -> None:
        if self.stuck_at_fault:
            mem_info = self.memristor_info_dict[self.device_name]
            SAF_lambda = mem_info['SAF_lambda']
            SAF_ratio = mem_info['SAF_ratio']
            SAF_delta = mem_info['SAF_delta']

            Q_ratio = self.SAF0_mask.float().mean() + self.SAF1_mask.float().mean()
            target_ratio = SAF_lambda + self.mem_t.max() * self.dt * SAF_delta
            increase_ratio = (target_ratio - Q_ratio) / (1 - Q_ratio)

            if increase_ratio > 0 and SAF_delta > 0:
                self.Q_mask.uniform_()
                self.SAF0_mask += (~(self.SAF0_mask + self.SAF1_mask)) & (self.Q_mask < ((SAF_ratio / (SAF_ratio + 1)) * increase_ratio))
                self.SAF1_mask += (~(self.SAF0_mask + self.SAF1_mask)) & \
                                  ((self.Q_mask >= ((SAF_ratio / (SAF_ratio + 1)) * increase_ratio)) & (self.Q_mask < increase_ratio))


    def total_energy_calculation(self) -> None:
        self.power.total_energy_calculation(mem_t=self.mem_t)

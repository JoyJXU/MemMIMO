from typing import Iterable, Optional, Union
from simbrain.memarray import MemristorArray
from simbrain.periphcircuit import DAC_Module
from simbrain.periphcircuit import ADC_Module
import json
import pickle
import torch
import os

SIMBRAIN_PATH = os.getenv("SIMBRAIN_PATH")

class Mapping(torch.nn.Module):
    # language=rst
    """
    Abstract base class for mapping neural networks to memristor arrays.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the layer.
        """
        super().__init__()

        self.sim_params = sim_params
        self.device_name = sim_params['device_name']
        self.device_structure = sim_params['device_structure']
        self.CMOS_technode = sim_params['CMOS_technode']
        self.device_roadmap = sim_params['device_roadmap']
        self.input_bit = sim_params['input_bit']
        self.ADC_setting = sim_params['ADC_setting']
        self.ADC_rounding_function = sim_params['ADC_rounding_function']

        if self.device_structure == 'trace':
            self.shape = [1, 1]  # Shape of the memristor crossbar
            for element in shape:
                self.shape[1] *= element
            self.shape = tuple(self.shape)
        elif self.device_structure in {'crossbar', 'mimo'}:
            self.shape = shape
        else:
            raise Exception("Only trace, mimo and crossbar architecture are supported!")
        
        self.register_buffer("mem_v", torch.Tensor())
        self.register_buffer("mem_x_read", torch.Tensor())
        self.register_buffer("mem_t", torch.Tensor())

        file_path = os.path.join(SIMBRAIN_PATH, 'memristor_device_info.json')
        with open(file_path, 'r') as f:
            self.memristor_info_dict = json.load(f)
        assert self.device_name in self.memristor_info_dict.keys(), "Invalid Memristor Device!"
        self.Gon = self.memristor_info_dict[self.device_name]['G_on']
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']
        self.v_read = self.memristor_info_dict[self.device_name]['v_read']

        file_path = os.path.join(SIMBRAIN_PATH, 'CMOS_tech_info.json')
        with open(file_path, 'r') as f:
            self.CMOS_tech_info_dict = json.load(f)
        assert self.device_roadmap in self.CMOS_tech_info_dict.keys(), "Invalid Memristor Device!"
        assert str(self.CMOS_technode) in self.CMOS_tech_info_dict[self.device_roadmap].keys(), "Invalid Memristor Device!"

        file_path = os.path.join(SIMBRAIN_PATH, 'memristor_lut.pkl')
        with open(file_path, 'rb') as f:
            self.memristor_luts = pickle.load(f)
        assert self.device_name in self.memristor_luts.keys(), "No Look-Up-Table Data Available for the Target Memristor Type!"

        self.trans_ratio = 1 / (self.Goff - self.Gon)

        self.batch_size = None
        self.learning = None

        self.sim_power = {}
        self.sim_periph_power = {}
        self.sim_area = {}


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when memristor is used to mapping traces.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.mem_v = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.mem_x_read = torch.zeros(batch_size, 1, self.shape[1], device=self.mem_x_read.device)
        self.mem_t = torch.zeros(batch_size, *self.shape, device=self.mem_t.device)


class MimoMapping(Mapping):
    # language=rst
    """
    Mapping MIMO to memristor arrays.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        """
        super().__init__(
            sim_params=sim_params,
            shape=shape
        )

        self.register_buffer("write_pulse_no", torch.Tensor())

        # Corssbar for positive input and positive weight
        self.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and positive weight
        self.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for positive input and negative weight
        self.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and negative weight
        self.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)

        self.DAC_module_pos = DAC_Module(sim_params=sim_params, shape=self.shape,
                                    CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        self.DAC_module_neg = DAC_Module(sim_params=sim_params, shape=self.shape,
                                    CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)

        if self.ADC_setting == 4:
            self.ADC_module_pos_pos = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_neg_pos = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_pos_neg = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_neg_neg = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        elif self.ADC_setting == 2:
            self.ADC_module_pos = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
            self.ADC_module_neg = ADC_Module(sim_params=sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")


        self.batch_interval = sim_params['batch_interval']


    def set_batch_size_mimo(self, batch_size) -> None:
        self.set_batch_size(batch_size)
        self.mem_pos_pos.set_batch_size(batch_size=batch_size)
        self.mem_neg_pos.set_batch_size(batch_size=batch_size)
        self.mem_pos_neg.set_batch_size(batch_size=batch_size)
        self.mem_neg_neg.set_batch_size(batch_size=batch_size)

        self.DAC_module_pos.set_batch_size(batch_size=batch_size)
        self.DAC_module_neg.set_batch_size(batch_size=batch_size)

        if self.ADC_setting == 4:
            self.ADC_module_pos_pos.set_batch_size(batch_size=batch_size)
            self.ADC_module_neg_pos.set_batch_size(batch_size=batch_size)
            self.ADC_module_pos_neg.set_batch_size(batch_size=batch_size)
            self.ADC_module_neg_neg.set_batch_size(batch_size=batch_size)
        elif self.ADC_setting == 2:
            self.ADC_module_pos.set_batch_size(batch_size=batch_size)
            self.ADC_module_neg.set_batch_size(batch_size=batch_size)
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        self.write_pulse_no = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)

        mem_t_matrix = (self.batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
        self.mem_t[:, :, :] = mem_t_matrix.view(-1, 1, 1)

        self.mem_pos_pos.mem_t = self.mem_t.clone()
        self.mem_neg_pos.mem_t = self.mem_t.clone()
        self.mem_pos_neg.mem_t = self.mem_t.clone()
        self.mem_neg_neg.mem_t = self.mem_t.clone()


    def mapping_write_mimo(self, target_x):
        # Memristor reset first
        v_reset = self.memristor_luts[self.device_name]['V_reset']
        self.mem_v.fill_(v_reset)
        # Adopt large negative pulses to reset the memristor array
        # self.mem_array.memristor_reset(mem_v=self.mem_v)
        self.DAC_module_pos.DAC_reset(mem_v=self.mem_v)
        self.DAC_module_neg.DAC_reset(mem_v=self.mem_v)

        self.mem_pos_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_pos_neg.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_neg.memristor_reset(mem_v=self.mem_v)

        total_wr_cycle = self.memristor_luts[self.device_name]['total_no']
        write_voltage = self.memristor_luts[self.device_name]['voltage']
        counter = torch.ones_like(self.mem_v)

        # Positive weight write
        matrix_pos = torch.relu(target_x)
        # Vector to Pulse Serial
        self.write_pulse_no = self.m2v(matrix_pos)
        # Matrix to memristor
        DAC_write_v = (self.write_pulse_no < (counter * total_wr_cycle)) * write_voltage
        self.DAC_module_pos.DAC_write(mem_v=DAC_write_v, mem_v_amp=write_voltage)
        # Memristor programming using multiple identical pulses (up to 400)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < self.write_pulse_no) * write_voltage
            self.mem_pos_pos.memristor_write(mem_v=self.mem_v)
            self.mem_neg_pos.memristor_write(mem_v=self.mem_v)

        # Negative weight write
        matrix_neg = torch.relu(target_x * -1)
        # Vector to Pulse Serial
        self.write_pulse_no = self.m2v(matrix_neg)
        # Matrix to memristor
        DAC_write_v = (self.write_pulse_no < (counter * total_wr_cycle)) * write_voltage
        self.DAC_module_neg.DAC_write(mem_v=DAC_write_v, mem_v_amp=write_voltage)
        # Memristor programming using multiple identical pulses (up to 400)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < self.write_pulse_no) * write_voltage
            self.mem_pos_neg.memristor_write(mem_v=self.mem_v)
            self.mem_neg_neg.memristor_write(mem_v=self.mem_v)


    def mapping_read_mimo(self, target_v):
        v_read_pos = self.DAC_module_pos.DAC_read(mem_v=target_v, sgn='pos')
        v_read_neg = self.DAC_module_neg.DAC_read(mem_v=target_v, sgn='neg')

        # memristor sequential read
        mem_i_sequence_pos_pos = self.mem_pos_pos.memristor_read(mem_v=v_read_pos)
        mem_i_sequence_neg_pos = self.mem_neg_pos.memristor_read(mem_v=v_read_neg)
        mem_i_sequence_pos_neg = self.mem_pos_neg.memristor_read(mem_v=v_read_pos)
        mem_i_sequence_neg_neg = self.mem_neg_neg.memristor_read(mem_v=v_read_neg)

        if self.ADC_setting == 4:
            mem_i_pos_pos = self.ADC_module_pos_pos.ADC_read(mem_i_sequence=mem_i_sequence_pos_pos, total_wire_resistance=self.mem_pos_pos.total_wire_resistance, high_cut_ratio=4/self.ADC_setting)
            mem_i_neg_pos = self.ADC_module_neg_pos.ADC_read(mem_i_sequence=mem_i_sequence_neg_pos, total_wire_resistance=self.mem_neg_pos.total_wire_resistance, high_cut_ratio=4/self.ADC_setting)
            mem_i_pos_neg = self.ADC_module_pos_neg.ADC_read(mem_i_sequence=mem_i_sequence_pos_neg, total_wire_resistance=self.mem_pos_neg.total_wire_resistance, high_cut_ratio=4/self.ADC_setting)
            mem_i_neg_neg = self.ADC_module_neg_neg.ADC_read(mem_i_sequence=mem_i_sequence_neg_neg, total_wire_resistance=self.mem_neg_neg.total_wire_resistance, high_cut_ratio=4/self.ADC_setting)
            mem_i = mem_i_pos_pos - mem_i_neg_pos - mem_i_pos_neg + mem_i_neg_neg
        elif self.ADC_setting == 2:
            mem_i_sequence_pos = mem_i_sequence_pos_pos + mem_i_sequence_neg_neg
            mem_i_pos = self.ADC_module_pos.ADC_read(mem_i_sequence_pos, total_wire_resistance=self.mem_pos_pos.total_wire_resistance, high_cut_ratio=4/self.ADC_setting)
            mem_i_sequence_neg = mem_i_sequence_neg_pos + mem_i_sequence_pos_neg
            mem_i_neg = self.ADC_module_pos.ADC_read(mem_i_sequence_neg, total_wire_resistance=self.mem_pos_neg.total_wire_resistance, high_cut_ratio=4/self.ADC_setting)
            mem_i = mem_i_pos - mem_i_neg
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        # Current to results
        if self.device_name == 'hu':
            self.mem_x_read = 1/(1.335e-9 - self.Gon) * mem_i / (2 ** self.input_bit - 1) / self.v_read
        else:
            self.mem_x_read = self.trans_ratio * mem_i / (2 ** self.input_bit - 1) / self.v_read
        return self.mem_x_read


    def m2v(self, target_matrix):
        # Target_matrix ranging [0, 1]
        within_range = (target_matrix >= 0) & (target_matrix <= 1)
        assert torch.all(within_range), "The target Matrix Must be in the Range [0, 1]!"

        # Target x to target conductance
        # target_c = target_matrix / self.trans_ratio + self.Gon

        # Get access to the look-up-table of the target memristor
        luts = self.memristor_luts[self.device_name]['conductance']
        luts = torch.tensor(luts, device=target_matrix.device)
        if self.device_name == 'hu':
            luts = (luts - self.Gon) / (1.335e-9 - self.Gon)
        else:
            luts = (luts - self.Gon) / (self.Goff - self.Gon)
        # Find the nearest conductance value
        c_diff = torch.abs(luts - target_matrix.unsqueeze(3))
        nearest_pulse_no = torch.argmin(c_diff, dim=3)

        return nearest_pulse_no


    def mem_t_update(self) -> None:
        self.mem_pos_pos.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_neg_pos.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_pos_neg.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_neg_neg.mem_t += self.batch_interval * (self.batch_size - 1)


    def total_energy_calculation(self) -> None:
        # language=rst
        """
        Calculate total energy for memristor-based architecture. Called when power is reported.
        """
        self.mem_pos_pos.total_energy_calculation()
        self.mem_neg_pos.total_energy_calculation()
        self.mem_pos_neg.total_energy_calculation()
        self.mem_neg_neg.total_energy_calculation()

        self.DAC_module_pos.DAC_energy_calculation(mem_t=self.mem_pos_pos.mem_t)
        self.DAC_module_neg.DAC_energy_calculation(mem_t=self.mem_neg_neg.mem_t)
        if self.ADC_setting == 4:
            self.ADC_module_pos_pos.ADC_energy_calculation(mem_t=self.mem_pos_pos.mem_t)
            self.ADC_module_neg_pos.ADC_energy_calculation(mem_t=self.mem_neg_pos.mem_t)
            self.ADC_module_pos_neg.ADC_energy_calculation(mem_t=self.mem_pos_neg.mem_t)
            self.ADC_module_neg_neg.ADC_energy_calculation(mem_t=self.mem_neg_neg.mem_t)
        elif self.ADC_setting == 2:
            self.ADC_module_pos.ADC_energy_calculation(mem_t=self.mem_pos_pos.mem_t)
            self.ADC_module_neg.ADC_energy_calculation(mem_t=self.mem_pos_neg.mem_t)
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        self.sim_power = {key: self.mem_pos_pos.power.sim_power[key] + self.mem_neg_pos.power.sim_power[key] +
                               self.mem_pos_neg.power.sim_power[key] + self.mem_neg_neg.power.sim_power[key] if key != 'time'
                          else self.mem_pos_pos.power.sim_power[key]
                          for key in self.mem_pos_pos.power.sim_power.keys()}

        self.sim_DAC_power = {key: self.DAC_module_pos.DAC_module_power.sim_power[key] + self.DAC_module_neg.DAC_module_power.sim_power[key]
                          for key in self.DAC_module_pos.DAC_module_power.sim_power.keys()}

        if self.ADC_setting == 4:
            self.sim_ADC_power = {key: self.ADC_module_pos_pos.ADC_module_power.sim_power[key] + self.ADC_module_neg_pos.ADC_module_power.sim_power[key] +
                                   self.ADC_module_pos_neg.ADC_module_power.sim_power[key] + self.ADC_module_neg_neg.ADC_module_power.sim_power[key]
                              for key in self.ADC_module_pos_pos.ADC_module_power.sim_power.keys()}
        elif self.ADC_setting == 2:
            self.sim_ADC_power = {key: self.ADC_module_pos.ADC_module_power.sim_power[key] + self.ADC_module_neg.ADC_module_power.sim_power[key]
                              for key in self.ADC_module_pos.ADC_module_power.sim_power.keys()}
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        self.sim_periph_power = {**self.sim_DAC_power, **self.sim_ADC_power}


    def total_area_calculation(self) -> None:
        # language=rst
        """
        Calculate total area for memristor-based architecture. Called when power is reported.
        """
        self.sim_mem_area = self.mem_pos_pos.area.array_area + self.mem_neg_pos.area.array_area + self.mem_pos_neg.area.array_area + self.mem_neg_neg.area.array_area

        DAC_height_row_pos, DAC_width_row_pos, DAC_height_col_pos, DAC_width_col_pos, sim_switch_matrix_row_area_pos, sim_switch_matrix_col_area_pos = self.DAC_module_pos.DAC_module_area.DAC_module_cal_area()
        DAC_height_row_neg, DAC_width_row_neg, DAC_height_col_neg, DAC_width_col_neg, sim_switch_matrix_row_area_neg, sim_switch_matrix_col_area_neg = self.DAC_module_neg.DAC_module_area.DAC_module_cal_area()
        sim_switch_matrix_row_area = sim_switch_matrix_row_area_neg + sim_switch_matrix_row_area_pos
        sim_switch_matrix_col_area = sim_switch_matrix_col_area_neg + sim_switch_matrix_col_area_pos

        if self.ADC_setting == 4:
            ADC_height_pos_pos, ADC_width_pos_pos, sim_shiftadd_area_pos_pos, sim_SarADC_area_pos_pos = self.ADC_module_pos_pos.ADC_module_area.ADC_module_cal_area()
            ADC_height_neg_pos, ADC_width_neg_pos, sim_shiftadd_area_neg_pos, sim_SarADC_area_neg_pos = self.ADC_module_neg_pos.ADC_module_area.ADC_module_cal_area()
            ADC_height_pos_neg, ADC_width_pos_neg, sim_shiftadd_area_pos_neg, sim_SarADC_area_pos_neg = self.ADC_module_pos_neg.ADC_module_area.ADC_module_cal_area()
            ADC_height_neg_neg, ADC_width_neg_neg, sim_shiftadd_area_neg_neg, sim_SarADC_area_neg_neg = self.ADC_module_neg_neg.ADC_module_area.ADC_module_cal_area()
            sim_SarADC_area = sim_SarADC_area_neg_neg + sim_SarADC_area_neg_pos + sim_SarADC_area_pos_neg + sim_SarADC_area_pos_pos
            sim_shiftadd_area = sim_shiftadd_area_neg_neg + sim_shiftadd_area_neg_pos + sim_shiftadd_area_pos_neg + sim_shiftadd_area_pos_pos
        elif self.ADC_setting == 2:
            ADC_height_pos, ADC_width_pos, sim_shiftadd_area_pos, sim_SarADC_area_pos = self.ADC_module_pos.ADC_module_area.ADC_module_cal_area()
            ADC_height_neg, ADC_width_neg, sim_shiftadd_area_neg, sim_SarADC_area_neg = self.ADC_module_neg.ADC_module_area.ADC_module_cal_area()
            sim_shiftadd_area = sim_shiftadd_area_neg + sim_shiftadd_area_pos
            sim_SarADC_area = sim_SarADC_area_neg + sim_SarADC_area_pos
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        periph_total_area = sim_switch_matrix_row_area + sim_switch_matrix_col_area + sim_shiftadd_area + sim_SarADC_area
        self.sim_periph_area = {'sim_switch_matrix_row_area': sim_switch_matrix_row_area,
                             'sim_switch_matrix_col_area': sim_switch_matrix_col_area,
                             'sim_shiftadd_area': sim_shiftadd_area, 'sim_SarADC_area': sim_SarADC_area,
                             'sim_total_periph_area': periph_total_area}

        if self.ADC_setting == 4:
            total_height_pos_pos = max(self.mem_pos_pos.length_col + ADC_height_pos_pos + DAC_height_col_pos, DAC_height_row_pos)
            total_width_pos_pos = DAC_width_row_pos + max(self.mem_pos_pos.length_row, DAC_width_col_pos, ADC_width_pos_pos)
            total_height_neg_pos = self.mem_neg_pos.length_col + ADC_height_neg_pos
            total_width_neg_pos = max(self.mem_neg_pos.length_row, ADC_width_neg_pos)
            total_height_pos_neg = max(self.mem_pos_neg.length_col + ADC_height_pos_neg + DAC_height_col_neg, DAC_height_row_neg)
            total_width_pos_neg = DAC_width_row_neg + max(self.mem_pos_neg.length_row, DAC_width_col_neg, ADC_width_pos_neg)
            total_height_neg_neg = self.mem_neg_neg.length_col + ADC_height_neg_neg
            total_width_neg_neg = max(self.mem_neg_neg.length_row, ADC_width_neg_neg)
        elif self.ADC_setting == 2:
            total_height_pos_pos = max(self.mem_pos_pos.length_col + ADC_height_pos + DAC_height_col_pos, DAC_height_row_pos)
            total_width_pos_pos = DAC_width_row_pos + max(self.mem_pos_pos.length_row, DAC_width_col_pos, ADC_width_pos)
            total_height_neg_pos = self.mem_neg_pos.length_col
            total_width_neg_pos = self.mem_neg_pos.length_row
            total_height_pos_neg = max(self.mem_pos_neg.length_col + ADC_height_neg + DAC_height_col_neg, DAC_height_row_neg)
            total_width_pos_neg = DAC_width_row_neg + max(self.mem_pos_neg.length_row, DAC_width_col_neg, ADC_width_neg)
            total_height_neg_neg = self.mem_neg_neg.length_col
            total_width_neg_neg = self.mem_neg_neg.length_row
        else:
            raise Exception("Only 2-set and 4-set ADC are supported!")

        self.sim_total_area = total_height_pos_pos * total_width_pos_pos + total_height_pos_neg * total_width_pos_neg \
                    + total_height_neg_pos * total_width_neg_pos + total_height_neg_neg * total_width_neg_neg

        self.sim_area = {'sim_mem_area':self.sim_mem_area,
                         'sim_periph_area':periph_total_area,
                         'sim_total_area':self.sim_total_area,
                         'sim_used_area_ratio':(self.sim_mem_area+periph_total_area)/self.sim_total_area}

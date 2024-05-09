import torch
import math
from typing import Iterable, Optional, Union
from simbrain.formula import Formula

class DAC_Module_Power(torch.nn.Module):
    # language=rst
    """
    Abstract base class for power estimation of memristor crossbar.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        CMOS_tech_info_dict: dict = {},
        memristor_info_dict: dict = {},
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
        self.sim_params = sim_params
        self.CMOS_technode = sim_params['CMOS_technode']
        self.CMOS_technode_meter = self.CMOS_technode * 1e-9
        self.device_roadmap = sim_params['device_roadmap']
        self.device_name = sim_params['device_name']
        self.input_bit = sim_params['input_bit']
        self.CMOS_tech_info_dict = CMOS_tech_info_dict
        self.memristor_info_dict = memristor_info_dict
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']
        self.dt = memristor_info_dict[self.device_name]['delta_t']
        self.vdd = self.CMOS_tech_info_dict[self.device_roadmap][str(self.CMOS_technode)]['vdd']
        self.relax_ratio_col = self.memristor_info_dict[self.device_name]['relax_ratio_col'] # Leave space for adjacent memristors
        self.relax_ratio_row = self.memristor_info_dict[self.device_name]['relax_ratio_row']
        self.read_v_amp = self.memristor_info_dict[self.device_name]['v_read']
        self.formula_function = Formula(sim_params=self.sim_params, shape=self.shape,
                                        CMOS_tech_info_dict=self.CMOS_tech_info_dict)

        self.pn_size_ratio = self.CMOS_tech_info_dict[self.device_roadmap][str(self.CMOS_technode)]['pn_size_ratio']
        self.MIN_NMOS_SIZE = self.CMOS_tech_info_dict['Constant']['MIN_NMOS_SIZE']
        self.MAX_TRANSISTOR_HEIGHT = self.CMOS_tech_info_dict['Constant']['MAX_TRANSISTOR_HEIGHT']
        self.IR_DROP_TOLERANCE = self.CMOS_tech_info_dict['Constant']['IR_DROP_TOLERANCE']
        self.POLY_WIDTH = self.CMOS_tech_info_dict['Constant']['POLY_WIDTH']
        self.MIN_GAP_BET_GATE_POLY = self.CMOS_tech_info_dict['Constant']['MIN_GAP_BET_GATE_POLY']

        self.switch_matrix_read_initialize()
        self.switch_matrix_col_write_initialize()
        self.switch_matrix_row_write_initialize()
        self.DFF_initialize()
        self.switch_matrix_reset_energy = 0
        self.DAC_total_energy = 0
        self.DAC_average_power = 0
        self.sim_power = {}


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size


    def switch_matrix_read_initialize(self) -> None:
        res_mem_cell_on_at_vw = 1 / self.Goff
        resTg = res_mem_cell_on_at_vw / self.shape[1] * self.IR_DROP_TOLERANCE
        min_cell_height = self.MAX_TRANSISTOR_HEIGHT * self.CMOS_technode_meter
        mem_size = self.memristor_info_dict[self.device_name]['mem_size'] * 1e-9
        length_col = self.shape[0] * self.relax_ratio_col * mem_size
        if length_col < min_cell_height:
            length_col = min_cell_height
        num_tg_pair_per_col = (int)(length_col / min_cell_height)
        num_col_tg_pair = (int)(math.ceil((float)(self.shape[0]) / num_tg_pair_per_col))
        num_tg_pair_per_col = (int)(math.ceil((float)(self.shape[0]) / num_col_tg_pair))
        switch_matrix_read_height = length_col / num_tg_pair_per_col
        switch_matrix_read_width_tg_N = self.formula_function.calculate_on_resistance(width=self.CMOS_technode_meter,
                                                                                      CMOS_type="NMOS") * self.CMOS_technode_meter / (resTg * 2)
        switch_matrix_read_width_tg_P = self.formula_function.calculate_on_resistance(width=self.CMOS_technode_meter,
                                                                                      CMOS_type="PMOS") * self.CMOS_technode_meter / (resTg * 2)
        self.switch_matrix_read_cap_gateN = self.formula_function.calculate_gate_cap(
            width=switch_matrix_read_width_tg_N)
        self.switch_matrix_read_cap_gateP = self.formula_function.calculate_gate_cap(
            width=switch_matrix_read_width_tg_P)
        _, self.switch_matrix_read_cap_tg_drain = self.formula_function.calculate_gate_capacitance(gateType='INV',
                                                                                                   numInput=1,
                                                                                                   widthNMOS=switch_matrix_read_width_tg_N,
                                                                                                   widthPMOS=switch_matrix_read_width_tg_P,
                                                                                                   heightTransistorRegion=switch_matrix_read_height)
        self.switch_matrix_read_energy = 0


    def switch_matrix_row_write_initialize(self) -> None:
        self.switch_matrix_row_write_cap_gateN = self.switch_matrix_read_cap_gateN
        self.switch_matrix_row_write_cap_gateP = self.switch_matrix_read_cap_gateP
        self.switch_matrix_row_write_cap_tg_drain = self.switch_matrix_read_cap_tg_drain
        self.switch_matrix_row_write_energy = 0


    def switch_matrix_col_write_initialize(self) -> None:
        num_row_tg_pair = 1
        min_cell_width = 2 * (self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter

        mem_size = self.memristor_info_dict[self.device_name]['mem_size'] * 1e-9
        length_row = self.shape[1] * self.relax_ratio_row * mem_size
        if length_row < min_cell_width * 2:
            length_row = min_cell_width * 2
        num_tg_pair_per_row = (int)(length_row / (min_cell_width * 2))
        num_row_tg_pair = (int)(math.ceil((float)(self.shape[1]) / num_tg_pair_per_row))
        num_tg_pair_per_row = (int)(math.ceil((float)(self.shape[1]) / num_row_tg_pair))
        tg_width = length_row / num_tg_pair_per_row / 2
        num_fold = (int)(tg_width / (0.5 * min_cell_width)) - 1
        res_mem_cell_on_at_vw = 1 / self.Goff
        resTg = res_mem_cell_on_at_vw / self.shape[0] / 2
        switch_matrix_col_write_width_tg_N = self.formula_function.calculate_on_resistance(
            width=self.CMOS_technode_meter, CMOS_type="NMOS") * self.CMOS_technode_meter / (resTg * 2)
        switch_matrix_col_write_width_tg_P = self.formula_function.calculate_on_resistance(
            width=self.CMOS_technode_meter, CMOS_type="PMOS") * self.CMOS_technode_meter / (resTg * 2)
        _, switch_matrix_col_write_height = self.formula_function.calculate_pass_gate_area(
            widthNMOS=switch_matrix_col_write_width_tg_N, widthPMOS=switch_matrix_col_write_width_tg_P,
            numFold=num_fold)
        self.switch_matrix_col_write_cap_gateN = self.formula_function.calculate_gate_cap(
            width=switch_matrix_col_write_width_tg_N)
        self.switch_matrix_col_write_cap_gateP = self.formula_function.calculate_gate_cap(
            width=switch_matrix_col_write_width_tg_P)
        _, self.switch_matrix_col_write_cap_tg_drain = self.formula_function.calculate_gate_capacitance(gateType='INV',
                                                                                                        numInput=1,
                                                                                                        widthNMOS=switch_matrix_col_write_width_tg_N,
                                                                                                        widthPMOS=switch_matrix_col_write_width_tg_P,
                                                                                                        heightTransistorRegion=switch_matrix_col_write_height)
        self.switch_matrix_col_write_energy = 0


    def DFF_initialize(self) -> None:
        DFF_width_inv_N = self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        DFF_width_inv_P = self.pn_size_ratio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        DFF_height_inv = self.MAX_TRANSISTOR_HEIGHT * self.CMOS_technode_meter
        DFF_width_tg_N = self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        DFF_width_tg_P = self.pn_size_ratio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        self.DFF_cap_inv_input, self.DFF_cap_inv_output = self.formula_function.calculate_gate_capacitance(
            gateType='INV', numInput=1, widthNMOS=DFF_width_inv_N, widthPMOS=DFF_width_inv_P, heightTransistorRegion=DFF_height_inv)
        self.DFF_cap_tg_gateN = self.formula_function.calculate_gate_cap(width=DFF_width_tg_N)
        self.DFF_cap_tg_gateP = self.formula_function.calculate_gate_cap(width=DFF_width_tg_P)
        _, self.DFF_cap_tg_drain = self.formula_function.calculate_gate_capacitance(gateType='INV',
                                                                                    numInput=1,
                                                                                    widthNMOS=DFF_width_tg_N,
                                                                                    widthPMOS=DFF_width_tg_P,
                                                                                    heightTransistorRegion=DFF_height_inv)
        self.DFF_energy = 0

    def switch_matrix_read_energy_calculation(self, activity_read, mem_v_shape) -> None:
        read_times = mem_v_shape[0] * mem_v_shape[1] * self.input_bit
        self.switch_matrix_read_energy += (self.switch_matrix_read_cap_tg_drain * 3) * self.read_v_amp * self.read_v_amp * activity_read * self.shape[0] * read_times
        self.switch_matrix_read_energy += (self.switch_matrix_read_cap_gateN + self.switch_matrix_read_cap_gateP) * self.vdd * self.vdd * activity_read * self.shape[0] * read_times
        self.switch_matrix_read_energy += self.DFF_energy_calculation(DFF_num=self.shape[0], DFF_read=read_times)


    def switch_matrix_col_write_energy_calculation(self, mem_v) -> None:
        mem_v_amp_pos = torch.max(mem_v)
        mem_v_amp_neg = torch.min(mem_v)
        mem_pos_num = (mem_v > 0).sum().item()
        mem_neg_num = (mem_v < 0).sum().item()
        self.switch_matrix_col_write_energy += (self.switch_matrix_col_write_cap_tg_drain * 3) * mem_v_amp_pos * mem_v_amp_pos * mem_pos_num
        self.switch_matrix_col_write_energy += (self.switch_matrix_col_write_cap_tg_drain * 3) * mem_v_amp_neg * mem_v_amp_neg * mem_neg_num
        self.switch_matrix_col_write_energy += (self.switch_matrix_col_write_cap_gateN + self.switch_matrix_col_write_cap_gateP) * self.vdd * self.vdd * (mem_neg_num + mem_pos_num)
        self.switch_matrix_col_write_energy += self.DFF_energy_calculation(self.shape[1], self.shape[0]*self.batch_size)


    def switch_matrix_row_write_energy_calculation(self, mem_v_amp) -> None:
        self.switch_matrix_row_write_energy += (self.switch_matrix_row_write_cap_tg_drain * 3) * 1/2 * mem_v_amp * 1/2 * mem_v_amp * self.batch_size * (self.shape[0]-1) * self.shape[0]
        self.switch_matrix_row_write_energy += (self.switch_matrix_row_write_cap_tg_drain * 3) * mem_v_amp * mem_v_amp * self.batch_size * self.shape[0]
        self.switch_matrix_row_write_energy += (self.switch_matrix_row_write_cap_gateN + self.switch_matrix_row_write_cap_gateP) * self.vdd * self.vdd * self.batch_size * self.shape[0] * self.shape[0]
        self.switch_matrix_row_write_energy += self.DFF_energy_calculation(self.shape[0], self.shape[0]*self.batch_size)


    def switch_matrix_reset_energy_calculation(self, mem_v) -> None:
        mem_v_amp = torch.min(mem_v)
        self.switch_matrix_reset_energy += (self.switch_matrix_col_write_cap_tg_drain * 3) * mem_v_amp * mem_v_amp * self.batch_size * self.shape[1]
        self.switch_matrix_reset_energy += (self.switch_matrix_col_write_cap_gateN + self.switch_matrix_col_write_cap_gateP) * self.vdd * self.vdd * self.batch_size * self.shape[1]
        self.switch_matrix_reset_energy += self.DFF_energy_calculation(self.shape[1], self.batch_size)


    def DFF_energy_calculation(self, DFF_num, DFF_read) -> None:
        self.DFF_energy = 0
        # Assume input D=1 and the energy of CLK INV and CLK TG are for 1 clock cycles
        # CLK INV (all DFFs have energy consumption)
        self.DFF_energy += (self.DFF_cap_inv_input + self.DFF_cap_inv_output) * self.vdd * self.vdd * 4 * DFF_num
        # CLK TG (all DFFs have energy consumption)
        self.DFF_energy += self.DFF_cap_tg_gateN * self.vdd * self.vdd * 2 * DFF_num
        self.DFF_energy += self.DFF_cap_tg_gateP * self.vdd * self.vdd * 2 * DFF_num
        # D to Q path (only selected DFFs have energy consumption)
        self.DFF_energy += (self.DFF_cap_tg_drain * 3 + self.DFF_cap_inv_input) * self.vdd * self.vdd * DFF_num
        self.DFF_energy += (self.DFF_cap_tg_drain + self.DFF_cap_inv_output) * self.vdd * self.vdd * DFF_num
        self.DFF_energy += (self.DFF_cap_inv_input + self.DFF_cap_inv_output) * self.vdd * self.vdd * DFF_num

        self.DFF_energy *= DFF_read
        return self.DFF_energy

    def DAC_energy_calculation(self, mem_t) -> None:
        # language=rst
        """
        Calculate total energy for memrisotr crossbar. Called when power is reported.

        :param mem_t: Time of the memristor crossbar.
        """
        self.DAC_total_energy = self.switch_matrix_col_write_energy + self.switch_matrix_row_write_energy + \
                                self.switch_matrix_read_energy + self.switch_matrix_reset_energy
        self.DAC_average_power = self.DAC_total_energy / (torch.max(mem_t) * self.dt)
        self.sim_power = {'switch_matrix_reset_energy': self.switch_matrix_reset_energy,
                          'switch_matrix_col_write_energy': self.switch_matrix_col_write_energy,
                          'switch_matrix_row_write_energy': self.switch_matrix_row_write_energy,
                          'switch_matrix_read_energy': self.switch_matrix_read_energy,
                          'DAC_total_energy': self.DAC_total_energy,
                          'DAC_average_power': self.DAC_average_power}


class ADC_Module_Power(torch.nn.Module):
    # language=rst
    """
    Abstract base class for power estimation of memristor crossbar.
    """
    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        CMOS_tech_info_dict: dict = {},
        memristor_info_dict: dict = {},
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
        self.sim_params = sim_params
        self.CMOS_technode = sim_params['CMOS_technode']
        self.CMOS_technode_meter = self.CMOS_technode * 1e-9
        self.device_roadmap = sim_params['device_roadmap']
        self.device_name = sim_params['device_name']
        self.temperature = sim_params['temperature']
        self.input_bit = sim_params['input_bit']
        self.ADC_precision = sim_params['ADC_precision']
        self.CMOS_tech_info_dict = CMOS_tech_info_dict
        self.memristor_info_dict = memristor_info_dict
        self.dt = memristor_info_dict[self.device_name]['delta_t']
        self.vdd = self.CMOS_tech_info_dict[self.device_roadmap][str(self.CMOS_technode)]['vdd']
        self.formula_function = Formula(sim_params=self.sim_params, shape=self.shape, CMOS_tech_info_dict=self.CMOS_tech_info_dict)

        self.pn_size_ratio = self.CMOS_tech_info_dict[self.device_roadmap][str(self.CMOS_technode)]['pn_size_ratio']
        self.MIN_NMOS_SIZE = self.CMOS_tech_info_dict['Constant']['MIN_NMOS_SIZE']
        self.MAX_TRANSISTOR_HEIGHT = self.CMOS_tech_info_dict['Constant']['MAX_TRANSISTOR_HEIGHT']

        self.DFF_initialize()
        self.adder_initialize()
        self.shift_add_energy = 0
        self.SarADC_energy = 0
        self.ADC_total_energy = 0
        self.ADC_average_power = 0
        self.sim_power = {}


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size


    def DFF_initialize(self) -> None:
        DFF_width_inv_N = self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        DFF_width_inv_P = self.pn_size_ratio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        DFF_height_inv =  self.MAX_TRANSISTOR_HEIGHT * self.CMOS_technode_meter
        DFF_width_tg_N = self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        DFF_width_tg_P = self.pn_size_ratio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        self.DFF_cap_inv_input, self.DFF_cap_inv_output = self.formula_function.calculate_gate_capacitance(gateType='INV', \
                                                        numInput=1, widthNMOS=DFF_width_inv_N, widthPMOS=DFF_width_inv_P, heightTransistorRegion=DFF_height_inv)
        self.DFF_cap_tg_gateN = self.formula_function.calculate_gate_cap(width=DFF_width_tg_N)
        self.DFF_cap_tg_gateP = self.formula_function.calculate_gate_cap(width=DFF_width_tg_P)
        _, self.DFF_cap_tg_drain = self.formula_function.calculate_gate_capacitance(gateType='INV', \
                                                        numInput=1, widthNMOS=DFF_width_tg_N, widthPMOS=DFF_width_tg_P, heightTransistorRegion=DFF_height_inv)
        self.DFF_energy = 0


    def adder_initialize(self) -> None:
        adder_width_nand_N = 2 * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        adder_width_nand_P = self.pn_size_ratio * self.MIN_NMOS_SIZE * self.CMOS_technode_meter
        adder_height_nand =  self.MAX_TRANSISTOR_HEIGHT * self.CMOS_technode_meter
        self.adder_cap_nand_input, self.adder_cap_nand_output = self.formula_function.calculate_gate_capacitance(gateType='NAND', \
                                                        numInput=2, widthNMOS=adder_width_nand_N, widthPMOS=adder_width_nand_P, heightTransistorRegion=adder_height_nand)
        self.adder_energy = 0


    def DFF_energy_calculation(self, DFF_num, DFF_read) -> None:
        self.DFF_energy = 0
        # Assume input D=1 and the energy of CLK INV and CLK TG are for 1 clock cycles
        # CLK INV (all DFFs have energy consumption)
        self.DFF_energy += (self.DFF_cap_inv_input + self.DFF_cap_inv_output) * self.vdd * self.vdd * 4 * DFF_num
		# CLK TG (all DFFs have energy consumption)
        self.DFF_energy += self.DFF_cap_tg_gateN * self.vdd * self.vdd * 2 * DFF_num
        self.DFF_energy += self.DFF_cap_tg_gateP * self.vdd * self.vdd * 2 * DFF_num
		# D to Q path (only selected DFFs have energy consumption)
        self.DFF_energy += (self.DFF_cap_tg_drain * 3 + self.DFF_cap_inv_input) * self.vdd * self.vdd * DFF_num
        self.DFF_energy += (self.DFF_cap_tg_drain  + self.DFF_cap_inv_output) * self.vdd * self.vdd * DFF_num
        self.DFF_energy += (self.DFF_cap_inv_input + self.DFF_cap_inv_output) * self.vdd * self.vdd * DFF_num

        self.DFF_energy *= DFF_read
        return self.DFF_energy

    def SarADC_energy_calculation(self, mem_i_sequence) -> None:
        # in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread (different from 0.5V)
        SarADC_energy_matrix = torch.zeros_like(mem_i_sequence)
        mem_r = torch.zeros_like(mem_i_sequence)
        mem_r = torch.where(mem_i_sequence != 0, 0.5 / mem_i_sequence, 1e15)
        mem_r = torch.where(mem_r != 0, mem_r, 1e-15)
        if self.device_roadmap == 'HP':
            if self.CMOS_technode == 130:
                SarADC_energy_matrix = (6.4806 * self.ADC_precision + 49.047) * 1e-6
                SarADC_energy_matrix += 0.207452 * torch.exp(-2.367 * torch.log10(mem_r))
            elif self.CMOS_technode == 90:
                SarADC_energy_matrix = (4.3474 * self.ADC_precision + 31.782) * 1e-6
                SarADC_energy_matrix += 0.164900 * torch.exp(-2.345 * torch.log10(mem_r))
            elif self.CMOS_technode == 65:
                SarADC_energy_matrix = (2.9503 * self.ADC_precision + 22.047) * 1e-6
                SarADC_energy_matrix += 0.128483 * torch.exp(-2.321 * torch.log10(mem_r))
            elif self.CMOS_technode == 45:
                SarADC_energy_matrix = (2.1843 * self.ADC_precision + 11.931) * 1e-6
                SarADC_energy_matrix += 0.097754 * torch.exp(-2.296 * torch.log10(mem_r))
            elif self.CMOS_technode == 32:
                SarADC_energy_matrix = (1.0157 * self.ADC_precision + 7.6286) * 1e-6
                SarADC_energy_matrix += 0.083709 * torch.exp(-2.313 * torch.log10(mem_r))
            elif self.CMOS_technode == 22:
                SarADC_energy_matrix = (0.7213 * self.ADC_precision + 3.3041) * 1e-6
                SarADC_energy_matrix += 0.084273 * torch.exp(-2.311 * torch.log10(mem_r))
            elif self.CMOS_technode == 14:
                SarADC_energy_matrix = (0.4710 * self.ADC_precision + 1.9529) * 1e-6
                SarADC_energy_matrix += 0.060584 * torch.exp(-2.311 * torch.log10(mem_r))
            elif self.CMOS_technode == 10:
                SarADC_energy_matrix = (0.3076 * self.ADC_precision + 1.1543) * 1e-6
                SarADC_energy_matrix += 0.049418 * torch.exp(-2.311 * torch.log10(mem_r))
            elif self.CMOS_technode == 7:
                SarADC_energy_matrix = (0.2008 * self.ADC_precision + 0.6823) * 1e-6
                SarADC_energy_matrix += 0.040310 * torch.exp(-2.311 * torch.log10(mem_r))
            else:
                raise Exception("Only limited CMOS technodes are supported!")
        elif self.device_roadmap == 'LP':
            if self.CMOS_technode == 130:
                SarADC_energy_matrix = (8.4483 * self.ADC_precision + 65.243) * 1e-6
                SarADC_energy_matrix += 0.169380 * torch.exp(-2.303 * torch.log10(mem_r))
            elif self.CMOS_technode == 90:
                SarADC_energy_matrix = (5.9869 * self.ADC_precision + 37.462) * 1e-6
                SarADC_energy_matrix += 0.144323 * torch.exp(-2.303 * torch.log10(mem_r))
            elif self.CMOS_technode == 65:
                SarADC_energy_matrix = (3.7506 * self.ADC_precision + 25.844) * 1e-6
                SarADC_energy_matrix += 0.121272 * torch.exp(-2.303 * torch.log10(mem_r))
            elif self.CMOS_technode == 45:
                SarADC_energy_matrix = (2.1691 * self.ADC_precision + 16.693) * 1e-6
                SarADC_energy_matrix += 0.100225 * torch.exp(-2.303 * torch.log10(mem_r))
            elif self.CMOS_technode == 32:
                SarADC_energy_matrix = (1.1294 * self.ADC_precision + 8.8998) * 1e-6
                SarADC_energy_matrix += 0.079449 * torch.exp(-2.297 * torch.log10(mem_r))
            elif self.CMOS_technode == 22:
                SarADC_energy_matrix = (0.538 * self.ADC_precision + 4.3753) * 1e-6
                SarADC_energy_matrix += 0.072341 * torch.exp(-2.303 * torch.log10(mem_r))
            elif self.CMOS_technode == 14:
                SarADC_energy_matrix = (0.3132 * self.ADC_precision + 2.5681) * 1e-6
                SarADC_energy_matrix += 0.061085 * torch.exp(-2.303 * torch.log10(mem_r))
            elif self.CMOS_technode == 10:
                SarADC_energy_matrix = (0.1823 * self.ADC_precision + 1.5073) * 1e-6
                SarADC_energy_matrix += 0.051580 * torch.exp(-2.303 * torch.log10(mem_r))
            elif self.CMOS_technode == 7:
                SarADC_energy_matrix = (0.1061 * self.ADC_precision + 0.8847) * 1e-6
                SarADC_energy_matrix += 0.043555 * torch.exp(-2.303 * torch.log10(mem_r))
            else:
                raise Exception("Only limited CMOS technodes are supported!")
        else:
            raise Exception("Only HP & LP are supported!")

        SarADC_energy_matrix = torch.where(mem_i_sequence != 0, SarADC_energy_matrix, 1e-6)
        SarADC_energy_matrix = torch.where(mem_r != 0, SarADC_energy_matrix, 0)

        SarADC_energy_matrix *= (1 + 1.3e-3 * (self.temperature - 300))
        SarADC_energy_matrix *= (self.ADC_precision + 1) * 1e-9
        self.SarADC_energy += torch.sum(SarADC_energy_matrix)


    def adder_energy_calculation(self, num_adder) -> None:
        self.adder_energy = 0
        self.adder_energy += (self.adder_cap_nand_input * 6) * self.vdd * self.vdd  # Input of 1 and 2 and Cin
        self.adder_energy += (self.adder_cap_nand_output * 2) * self.vdd * self.vdd  # Output of S[0] and 5
        # Second and later stages
        self.adder_energy += (self.adder_cap_nand_input * 7) * self.vdd * self.vdd * (self.ADC_precision - 1)
        self.adder_energy += (self.adder_cap_nand_output * 3) * self.vdd * self.vdd * (self.ADC_precision - 1)

        # Hidden transition
        # First stage
        self.adder_energy += (self.adder_cap_nand_output + self.adder_cap_nand_input) * self.vdd * self.vdd * 2	# #2 and #3
        self.adder_energy += (self.adder_cap_nand_output + self.adder_cap_nand_input * 2) * self.vdd * self.vdd	# #4
        self.adder_energy += (self.adder_cap_nand_output + self.adder_cap_nand_input * 3) * self.vdd * self.vdd	# #5
        self.adder_energy += (self.adder_cap_nand_output + self.adder_cap_nand_input) * self.vdd * self.vdd		# #6
        # Second and later stages
        self.adder_energy += (self.adder_cap_nand_output + self.adder_cap_nand_input * 3) * self.vdd * self.vdd * (
                    self.ADC_precision - 1)  # #1
        self.adder_energy += (self.adder_cap_nand_output + self.adder_cap_nand_input) * self.vdd * self.vdd * (
                    self.ADC_precision - 1)  # #3
        self.adder_energy += (self.adder_cap_nand_output + self.adder_cap_nand_input) * self.vdd * self.vdd * 2 * (
                    self.ADC_precision - 1)  # #6 and #7

        self.adder_energy *= self.input_bit * num_adder

        return self.adder_energy


    def shift_add_energy_calculation(self, mem_i_sequence) -> None:
        read_times = mem_i_sequence.shape[1] * mem_i_sequence.shape[2]
        self.shift_add_energy += self.DFF_energy_calculation(
            mem_i_sequence.shape[3] * (self.ADC_precision + self.input_bit), self.input_bit) * read_times
        self.shift_add_energy += self.adder_energy_calculation(mem_i_sequence.shape[3]) * read_times


    def ADC_energy_calculation(self, mem_t) -> None:
        # language=rst
        """
        Calculate total energy for memrisotr crossbar. Called when power is reported.

        :param mem_t: Time of the memristor crossbar.
        """
        self.ADC_total_energy = self.shift_add_energy + self.SarADC_energy
        self.ADC_average_power = self.ADC_total_energy / (torch.max(mem_t) * self.dt)
        self.sim_power = {'shift_add_energy': self.shift_add_energy,
                          'SarADC_energy': self.SarADC_energy,
                          'ADC_total_energy': self.ADC_total_energy,
                          'ADC_average_power': self.ADC_average_power}
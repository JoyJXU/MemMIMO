import torch
from typing import Iterable, Optional, Union
import math
import json


class Formula(torch.nn.Module):
    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        CMOS_tech_info_dict: dict = {},
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
        self.device_roadmap = sim_params['device_roadmap']
        self.CMOS_technode = sim_params['CMOS_technode']
        self.CMOS_technode_str = str(sim_params['CMOS_technode'])
        self.CMOS_technode_meter = self.CMOS_technode * 1e-9
        self.CMOS_tech_info_dict = CMOS_tech_info_dict
        self.temperature = sim_params['temperature']
        self.tempIndex = self.temperature - 300
        self.tempIndex_str = str(self.tempIndex)

        if self.CMOS_technode < 22:
            self.heightFin = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['heightFin']
            self.widthFin = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['widthFin']
            self.PitchFin = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['PitchFin']
        self.MIN_POLY_EXT_DIFF = self.CMOS_tech_info_dict['Constant']['MIN_POLY_EXT_DIFF']
        self.MIN_GAP_BET_FIELD_POLY = self.CMOS_tech_info_dict['Constant']['MIN_GAP_BET_FIELD_POLY']
        self.MIN_GAP_BET_P_AND_N_DIFFS = self.CMOS_tech_info_dict['Constant']['MIN_GAP_BET_P_AND_N_DIFFS']
        self.POLY_WIDTH = self.CMOS_tech_info_dict['Constant']['POLY_WIDTH']
        self.MIN_GAP_BET_GATE_POLY = self.CMOS_tech_info_dict['Constant']['MIN_GAP_BET_GATE_POLY']
        self.cap_ideal_gate = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['cap_ideal_gate']
        self.cap_overlap = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['cap_overlap']
        self.cap_fringe = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['cap_fringe']
        self.phy_gate_length = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['phy_gate_length']
        self.cap_polywire = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['cap_polywire']
        self.cap_junction = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['cap_junction']
        self.cap_sidewall = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['cap_sidewall']
        self.cap_drain_to_channel = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str][
            'cap_drain_to_channel']
        self.effective_resistance_multiplier = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str][
            'effective_resistance_multiplier']
        self.vdd = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['vdd']
        self.currentOnNmos = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['current_on_Nmos'][
            self.tempIndex_str]
        self.currentOnPmos = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode_str]['current_on_Pmos'][
            self.tempIndex_str]


    def calculate_gate_cap(self, width):
        return (self.cap_ideal_gate + self.cap_overlap + self.cap_fringe) * width + self.phy_gate_length * self.cap_polywire


    def calculate_gate_area(self, gateType, numInput, widthNMOS, widthPMOS, heightTransistorRegion):
        ratio = widthPMOS / (widthPMOS + widthNMOS)
        numFoldedPMOS = 1
        numFoldedNMOS = 1
        if self.CMOS_technode >= 22:  # Bulk
            if ratio == 0:  # no PMOS
                maxWidthPMOS = 0
                maxWidthNMOS = heightTransistorRegion - (
                            self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter
            elif ratio == 1:  # no NMOS
                maxWidthPMOS = heightTransistorRegion - (
                            self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter
                maxWidthNMOS = 0
            else:
                maxWidthPMOS = ratio * (
                            heightTransistorRegion - self.MIN_GAP_BET_P_AND_N_DIFFS * self.CMOS_technode_meter - (
                                self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter)
                maxWidthNMOS = maxWidthPMOS / ratio * (1 - ratio)

            if widthPMOS > 0:
                if widthPMOS <= maxWidthPMOS:  # No folding
                    unitWidthRegionP = 2 * (self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
                    heightRegionP = widthPMOS
                else:  # Folding
                    numFoldedPMOS = math.ceil(widthPMOS / maxWidthPMOS)
                    unitWidthRegionP = (numFoldedPMOS + 1) * (
                                self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
                    heightRegionP = maxWidthPMOS
            else:
                unitWidthRegionP = 0
                heightRegionP = 0

            if widthNMOS > 0:
                if widthNMOS <= maxWidthNMOS:  # No folding
                    unitWidthRegionN = 2 * (self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
                    heightRegionN = widthNMOS
                else:  # Folding
                    numFoldedNMOS = math.ceil(widthNMOS / maxWidthNMOS)
                    unitWidthRegionN = (numFoldedNMOS + 1) * (
                                self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
                    heightRegionN = maxWidthNMOS
            else:
                unitWidthRegionN = 0
                heightRegionN = 0

        else:  # FinFET
            if ratio == 0:  # no PFinFET
                maxNumPFin = 0
                maxNumNFin = math.floor((heightTransistorRegion - (
                            self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter) / self.PitchFin) + 1
            elif ratio == 1:  # no NFinFET
                maxNumPFin = math.floor((heightTransistorRegion - (
                            self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter) / self.PitchFin) + 1
                maxNumNFin = 0
            else:
                maxNumPFin = math.floor(ratio * (
                            heightTransistorRegion - self.MIN_GAP_BET_P_AND_N_DIFFS * self.CMOS_technode_meter - (
                                self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter) / self.PitchFin) + 1
                maxNumNFin = math.floor((1 - ratio) * (
                            heightTransistorRegion - self.MIN_GAP_BET_P_AND_N_DIFFS * self.CMOS_technode_meter - (
                                self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter) / self.PitchFin) + 1

            NumPFin = math.ceil(widthPMOS / (2 * self.heightFin + self.widthFin))
            if NumPFin > 0:
                if NumPFin <= maxNumPFin:  # No folding
                    unitWidthRegionP = 2 * (self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
                    heightRegionP = (NumPFin - 1) * self.PitchFin + 2 * self.widthFin / 2
                else:  # Folding
                    numFoldedPMOS = math.ceil(NumPFin / maxNumPFin)
                    unitWidthRegionP = (numFoldedPMOS + 1) * (
                                self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
                    heightRegionP = (maxNumPFin - 1) * self.PitchFin + 2 * self.widthFin / 2
            else:
                unitWidthRegionP = 0
                heightRegionP = 0

            NumNFin = math.ceil(widthNMOS / (2 * self.heightFin + self.widthFin))
            if NumNFin > 0:
                if NumNFin <= maxNumNFin:  # No folding
                    unitWidthRegionN = 2 * (self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
                    heightRegionN = (NumNFin - 1) * self.PitchFin + 2 * self.widthFin / 2
                else:  # Folding
                    numFoldedNMOS = math.ceil(NumNFin / maxNumNFin)
                    unitWidthRegionN = (numFoldedNMOS + 1) * (
                                self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
                    heightRegionN = (maxNumNFin - 1) * self.PitchFin + 2 * self.widthFin / 2
            else:
                unitWidthRegionN = 0
                heightRegionN = 0

        if gateType == 'INV':
            widthRegionP = unitWidthRegionP
            widthRegionN = unitWidthRegionN
        elif gateType in ['NOR', 'NAND']:
            if numFoldedPMOS == 1 and numFoldedNMOS == 1:  # Need to subtract the source/drain sharing region
                widthRegionP = unitWidthRegionP * numInput - (numInput - 1) * self.CMOS_technode_meter * (
                            self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY)
                widthRegionN = unitWidthRegionN * numInput - (numInput - 1) * self.CMOS_technode_meter * (
                            self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY)
        else:
            widthRegionN = 0
            widthRegionP = 0

        width = max(widthRegionN, widthRegionP)
        height = heightTransistorRegion
        return width, height


    def calculate_gate_capacitance(self, gateType, numInput, widthNMOS, widthPMOS, heightTransistorRegion):
        ratio = widthPMOS / (widthPMOS + widthNMOS)
        numFoldedPMOS = 1
        numFoldedNMOS = 1
        if self.CMOS_technode >= 22:  # Bulk
            if ratio == 0:  # no PMOS
                maxWidthPMOS = 0
                maxWidthNMOS = heightTransistorRegion - (
                            self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter
            elif ratio == 1:  # no NMOS
                maxWidthPMOS = heightTransistorRegion - (
                            self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter
                maxWidthNMOS = 0
            else:
                maxWidthPMOS = ratio * (
                        heightTransistorRegion - self.MIN_GAP_BET_P_AND_N_DIFFS * self.CMOS_technode_meter - (
                        self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter)
                maxWidthNMOS = maxWidthPMOS / ratio * (1 - ratio)

            if widthPMOS > 0:
                if widthPMOS <= maxWidthPMOS:  # No folding
                    UnitWidthDrainP = self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    UnitWidthSourceP = UnitWidthDrainP
                    heightDrainP = widthPMOS
                else:  # Folding
                    numFoldedPMOS = int(math.ceil(widthPMOS / maxWidthPMOS))
                    UnitWidthDrainP = int(
                        math.ceil((numFoldedPMOS + 1) / 2)) * self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    UnitWidthSourceP = int(
                        math.floor((numFoldedPMOS + 1) / 2)) * self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    heightDrainP = maxWidthPMOS
            else:
                UnitWidthDrainP = 0
                UnitWidthSourceP = 0
                heightDrainP = 0

            if widthNMOS > 0:
                if widthNMOS <= maxWidthNMOS:  # No folding
                    UnitWidthDrainN = self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    UnitWidthSourceN = UnitWidthDrainN
                    heightDrainN = widthNMOS
                else:  # Folding
                    numFoldedNMOS = int(math.ceil(widthNMOS / maxWidthNMOS))
                    UnitWidthDrainN = int(
                        math.ceil((numFoldedNMOS + 1) / 2)) * self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    UnitWidthSourceN = int(
                        math.floor((numFoldedNMOS + 1) / 2)) * self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    heightDrainN = maxWidthNMOS
            else:
                UnitWidthDrainN = 0
                UnitWidthSourceN = 0
                heightDrainN = 0

        else:  # FinFET
            if ratio == 0:  # no PFinFET
                maxNumPFin = 0
                maxNumNFin = int(math.floor(
                    (heightTransistorRegion - (
                                self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter) /
                    self.PitchFin)) + 1
            elif ratio == 1:  # no NFinFET
                maxNumPFin = int(math.floor(
                    (heightTransistorRegion - (
                                self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter) /
                    self.PitchFin)) + 1
                maxNumNFin = 0
            else:
                maxNumPFin = int(math.floor(
                    ratio * (heightTransistorRegion - self.MIN_GAP_BET_P_AND_N_DIFFS * self.CMOS_technode_meter - (
                            self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter) / self.PitchFin)) + 1
                maxNumNFin = int(math.floor((1 - ratio) * (
                        heightTransistorRegion - self.MIN_GAP_BET_P_AND_N_DIFFS * self.CMOS_technode_meter - (
                        self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter) / self.PitchFin)) + 1

            NumPFin = int(math.ceil(widthPMOS / (2 * self.heightFin + self.widthFin)))

            if NumPFin > 0:
                if NumPFin <= maxNumPFin:  # No folding
                    UnitWidthDrainP = self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    UnitWidthSourceP = UnitWidthDrainP
                    heightDrainP = (NumPFin - 1) * self.PitchFin + 2 * self.widthFin / 2
                else:  # Folding
                    numFoldedPMOS = int(math.ceil(NumPFin / maxNumPFin))
                    UnitWidthDrainP = int(
                        math.ceil((numFoldedPMOS + 1) / 2)) * self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    UnitWidthSourceP = int(
                        math.floor((numFoldedPMOS + 1) / 2)) * self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    heightDrainP = (maxNumPFin - 1) * self.PitchFin + 2 * self.widthFin / 2
            else:
                UnitWidthDrainP = 0
                UnitWidthSourceP = 0
                heightDrainP = 0

            NumNFin = int(math.ceil(widthNMOS / (2 * self.heightFin + self.widthFin)))

            if NumNFin > 0:
                if NumNFin <= maxNumNFin:  # No folding
                    UnitWidthDrainN = self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    UnitWidthSourceN = UnitWidthDrainN
                    heightDrainN = (NumNFin - 1) * self.PitchFin + 2 * self.widthFin / 2
                else:  # Folding
                    numFoldedNMOS = int(math.ceil(NumNFin / maxNumNFin))
                    UnitWidthDrainN = int(
                        math.ceil((numFoldedNMOS + 1) / 2)) * self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    UnitWidthSourceN = int(
                        math.floor((numFoldedNMOS + 1) / 2)) * self.CMOS_technode_meter * self.MIN_GAP_BET_GATE_POLY
                    heightDrainN = (maxNumNFin - 1) * self.PitchFin + 2 * self.widthFin / 2
            else:
                UnitWidthDrainN = 0
                UnitWidthSourceN = 0
                heightDrainN = 0

        # Switch case for gate type
        if gateType == "INV":
            if widthPMOS > 0:
                widthDrainP = UnitWidthDrainP
                widthDrainSidewallP = widthDrainP * 2 + heightDrainP * (1 + (numFoldedPMOS + 1) % 2)
            if widthNMOS > 0:
                widthDrainN = UnitWidthDrainN
                widthDrainSidewallN = widthDrainN * 2 + heightDrainN * (1 + (numFoldedPMOS + 1) % 2)

        elif gateType == "NOR":
            if numFoldedPMOS == 1 and numFoldedNMOS == 1:
                if widthPMOS > 0:
                    widthDrainP = UnitWidthDrainP * numInput
                    widthDrainSidewallP = widthDrainP * 2 + heightDrainP
                if widthNMOS > 0:
                    widthDrainN = UnitWidthDrainN * int(math.floor((numInput + 1) / 2))
                    widthDrainSidewallN = widthDrainN * 2 + heightDrainN * (1 - (numInput + 1) % 2)
            else:
                if widthPMOS > 0:
                    widthDrainP = UnitWidthDrainP * numInput + (numInput - 1) * UnitWidthSourceP
                    widthDrainSidewallP = widthDrainP * 2 + heightDrainP * (1 + (numFoldedPMOS + 1) % 2) * numInput \
                                          + heightDrainP * (1 - (numFoldedPMOS + 1) % 2) * (numInput - 1)
                if widthNMOS > 0:
                    widthDrainN = UnitWidthDrainN * numInput
                    widthDrainSidewallN = widthDrainN * 2 + heightDrainN * (1 + (numFoldedNMOS + 1) % 2) * numInput

        elif gateType == "NAND":
            if numFoldedPMOS == 1 and numFoldedNMOS == 1:
                if widthPMOS > 0:
                    widthDrainP = UnitWidthDrainP * int(math.floor((numInput + 1) / 2))
                    widthDrainSidewallP = widthDrainP * 2 + heightDrainP * (1 - (numInput + 1) % 2)
                if widthNMOS > 0:
                    widthDrainN = UnitWidthDrainN * numInput
                    widthDrainSidewallN = widthDrainN * 2 + heightDrainN
            else:
                if widthPMOS > 0:
                    widthDrainP = UnitWidthDrainP * numInput
                    widthDrainSidewallP = widthDrainP * 2 + heightDrainP * (1 + (numFoldedPMOS + 1) % 2) * numInput
                if widthNMOS > 0:
                    widthDrainN = UnitWidthDrainN * numInput + (numInput - 1) * UnitWidthSourceN
                    widthDrainSidewallN = widthDrainN * 2 + heightDrainN * (1 + (numFoldedNMOS + 1) % 2) * numInput \
                                          + heightDrainN * (1 - (numFoldedNMOS + 1) % 2) * (numInput - 1)

        else:
            widthDrainN = widthDrainP = widthDrainSidewallP = widthDrainSidewallN = 0

        # Junction capacitance
        cap_drain_bottom_n = 4 * heightDrainN * self.cap_junction
        cap_drain_bottom_p = widthDrainP * heightDrainP * self.cap_junction

        # Sidewall capacitance
        cap_drain_sidewall_n = widthDrainSidewallN * self.cap_sidewall
        cap_drain_sidewall_p = widthDrainSidewallP * self.cap_sidewall

        # Drain to channel capacitance
        cap_drain_to_channel_n = numFoldedNMOS * heightDrainN * self.cap_drain_to_channel
        cap_drain_to_channel_p = numFoldedPMOS * heightDrainP * self.cap_drain_to_channel

        cap_output = cap_drain_bottom_n + cap_drain_bottom_p + cap_drain_sidewall_n + cap_drain_sidewall_p + cap_drain_to_channel_n + cap_drain_to_channel_p
        cap_input = self.calculate_gate_cap(widthNMOS) + self.calculate_gate_cap(widthPMOS)

        return cap_input, cap_output


    def calculate_on_resistance(self, width, CMOS_type):
        if self.tempIndex > 100 or self.tempIndex < 0:
            print("Error: Temperature is out of range")
            exit(-1)

        if CMOS_type == "NMOS":
            r = self.effective_resistance_multiplier * self.vdd / (self.currentOnNmos * width)
        elif CMOS_type == "PMOS":
            r = self.effective_resistance_multiplier * self.vdd / (self.currentOnPmos * width)
        else:
            raise Exception("Only NMOS & PMOS are supported!")
        return r


    def calculate_pass_gate_area(self, widthNMOS, widthPMOS, numFold):
        if self.CMOS_technode >= 22:  # Bulk
            width = (numFold + 1) * (self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
            height = widthPMOS / numFold + widthNMOS / numFold + self.MIN_GAP_BET_P_AND_N_DIFFS * self.CMOS_technode_meter + (
                    self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter
        else:  # FinFET
            totalNumPFin = int(math.ceil(widthPMOS / (2 * self.heightFin + self.widthFin)))
            totalNumNFin = int(math.ceil(widthNMOS / (2 * self.heightFin + self.widthFin)))
            NumPFin = int(math.ceil(totalNumPFin / numFold))
            NumNFin = int(math.ceil(totalNumNFin / numFold))
            heightRegionP = (NumPFin - 1) * self.PitchFin + 2 * self.widthFin / 2
            heightRegionN = (NumNFin - 1) * self.PitchFin + 2 * self.widthFin / 2
            width = (numFold + 1) * (self.POLY_WIDTH + self.MIN_GAP_BET_GATE_POLY) * self.CMOS_technode_meter
            height = heightRegionP + heightRegionN + self.MIN_GAP_BET_P_AND_N_DIFFS * self.CMOS_technode_meter + (
                    self.MIN_POLY_EXT_DIFF + self.MIN_GAP_BET_FIELD_POLY / 2) * 2 * self.CMOS_technode_meter
        return width, height
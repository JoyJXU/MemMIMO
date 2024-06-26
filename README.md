# MemMIMO

The proposed Memristor_MIMO simulation framework integrates a behavioral model (MemMIMO) of the mixed-signal architecture with a digital front-end. MemMIMO is the behavioral model of the mixed-signal architecture based on memristors runninig in a Python environment.

The hardware performance estimation is based on the open-source RRAM add-on process library [NeuroSim](https://github.com/neurosim/MLP_NeuroSim_V3.0.git), with relevant functions re-implemented in Python.

# Requirements
* python
* cuda
* pytorch
* matplotlib

# How to Run
We provide a simple demo of simulate the complex MVM in the MIMO precoding algorithms to the crossbar-based architecture in examples/MIMO/main_test.py

```
git clone https://github.com/JoyJXU/MemMIMO.git
cd examples/MIMO/
python main_test.py
```

MemMIMO takes four different types of input, including the memristor data, matrix and vector data, MIMO system requirements, and simulation parameters.

## Memristor data:
Three memristors, including ferro, hu(FS), and MF memristor, are modeled with the parameters listed in memristor_device_info.json

You can use your own memristors by creating a new entry in the json file.

## Matrix and vector data:

The matrix and vector data are randomly generized in examples/MIMO/testbenches.py

In the demo, we are now using uniformly distribued numbers ranging from -1 to 1.

## MIMO system requirements:

The system requirements are extracted from use cases and serve as the experimental settings for the simulation framework. The system requirements include size of the input vector, size of the matrix, operation / update ratio, and the number of matrix updates.

The size of the input vector and the matrix can be set by indicating the '--cols' and '--rows' in examples/MIMO/main_test.py

The operation / update ratio can be adjusted by the '--read_batch' in examples/MIMO/main_test.py

The number of matrix updates can be adjusted by the '--rep' in examples/MIMO/main_test.py

## Simulation parameters:

The simulation parameters include the parameters for the memristor crossbar and the parameters for the peripheral circuit, and they can be set in examples/MIMO/main_test.py

| Parameters | Definition | Default | 
|:------------|:--------------:|:--------------:|
| --memristor_device          | Chose the memristor model, three are provided here: 'ferro', 'hu'(FS), and 'MF'         | 'MF'|
| --c2c_variation            | Whether to include the cycle-to-cycle variation in the simulation: True, False           | False|
| --d2d_variation            | Whether to include the device-to-device variation in the simulation: 0, 1, 2, 3. 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only         |0|
| --input_bit                | The DAC resolution: int (4-32)                                                            |8|
| --ADC_precision            | The ADC resolution: int (4-32)                                                            |16|
| --ADC_setting              | 2 or 4. Employing four sets equips each memristor crossbar with a dedicated ADC. Using two sets integrates two crossbars vertically due to their summable currents per column, allowing them to share a single ADC set.|4|
| --ADC_rounding_function    | 'floor' or 'round'                                                                        |'floor'|
| --wire_width              | In practice, the wire width shall be set around 1/2 of the memristor size: hu/MF - 10000 (10 um), ferro - 200 (200nm)|10000|
| --CMOS_technode            | Technology node for the peripheral circuits: 130, 90, 65, 45, 32, 22, 14, 10, 7 (nm)      |45|
| --device_roadmap            | High performance or low power: 'HP', 'LP'                                                |'HP'|
| --temperature	              | Default to 300 (K)                                                                |300|
| --hardware_estimation      | Whether to run hardware estimation: True, False|True|

# Examples
Using MF memristor in use case 1, without c2c or d2d variation
```
python main_test.py
```

Using MF memristor in use case 2, without c2c or d2d variation
```
python main_test.py --rows 48 --cols 512 --read_batch 336
```

Using FS memristor in use case 1, without c2c or d2d variation
```
python main_test.py --memristor_device 'hu'
```
				
Using ferroelectric memristor in use case 1, without c2c or d2d variation
```
python main_test.py --memristor_device 'ferro' --wire_width 200
```
				 				
Using MF memristor in use case 1, with c2c, without d2d variation
```
python main_test.py --c2c_variation True
```					
			
Using MF memristor in use case 1, with c2c, without d2d variation
```
python main_test.py --c2c_variation True
```					

Using MF memristor in use case 1, without c2c, with d2d variation in both Gon/Goff and nonlinearity
```
python main_test.py --d2d_variation 1
```					
			       
# Citation
If you find MemMIMO useful or relevant to your research, please kindly cite our paper:
```
@inproceedings{memmimo,
    title={MemMIMO: A Simulation Framework for Memristor-Based Massive MIMO Acceleration},
    author={Jiawei Xu, Yi Zheng, Dimitrios Stathis, Ruijia Wang, Ruisi Shen, Li-Rong Zheng, Zhuo Zou, Ahmed Hemani},
    year={2024}
}
```				
			
			
			
		

python plot_data.py noshow
cd ../data_firing_response/
python produce_reduced_data.py noshow
cd ../input_impedance_calibration/
python minimization.py compute
python minimization.py single_comp
python minimization.py noshow
python produce_all_cell_params.py

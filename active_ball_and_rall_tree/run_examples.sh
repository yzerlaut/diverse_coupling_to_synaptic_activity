# baseline
pyth2 demo.py -f data/baseline.npz --mean_model --tstop 2601
python demo.py -f data/baseline.npz -u
plot_to_svg data/baseline.npz baseline.svg
# baseline + stim
pyth2 demo.py -f data/stim_baseline.npz --mean_model --tstop 2601 --with_synch_stim --DT_syn_stim 500.
python demo.py -f data/stim_baseline.npz -u
plot_to_svg data/stim_baseline.npz stim_baseline.svg

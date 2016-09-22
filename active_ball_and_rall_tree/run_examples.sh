source ~/.bash_profile
# === baseline ===
pyth2 demo.py -f data/baseline.npz --mean_model --tstop 2601
python demo.py -f data/baseline.npz -u
plot_to_svg data/baseline.npz baseline.svg
# === baseline + stim ===
pyth2 demo.py -f data/stim_baseline.npz --mean_model --tstop 2601 --with_synch_stim --DT_synch_stim 500.
python demo.py -f data/stim_baseline.npz -u
plot_to_svg data/stim_baseline.npz stim_baseline.svg
# === proximal stim ===
pyth2 demo.py -f data/prox.npz --mean_model --tstop 2601 --fe_prox 1.1 --fi_prox 9.
python demo.py -f data/prox.npz -u
plot_to_svg data/prox.npz prox.svg
# === distal stim ===
pyth2 demo.py -f data/dist.npz --mean_model --tstop 2601 --fe_dist 0.6 --fi_dist 3.
python demo.py -f data/dist.npz -u
plot_to_svg data/dist.npz dist.svg
# === synchrony stim ===
pyth2 demo.py -f data/synch.npz --mean_model --tstop 2601 --synchrony 0.55
python demo.py -f data/synch.npz -u
plot_to_svg data/synch.npz synch.svg
# === unbalanced stim ===
pyth2 demo.py -f data/unbalanced.npz --mean_model --tstop 2601 --fe_dist 0.4 --fi_dist 1.5 --fe_prox 0.3 --fi_prox 1.
python demo.py -f data/unbalanced.npz -u
plot_to_svg data/unbalanced.npz unbalanced.svg


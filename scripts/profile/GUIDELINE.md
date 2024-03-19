srun -p EM --gres=gpu:4 rocprof --stats -o ./profile/rocprof_res.csv -i rocprof_roofline.txt ./build/apps/llama

# Create virtual venv with conda
pip install -r requirements-roofline.txt

python plot_roofline_hierarchical.py rocprof_res04.csv roofline.png

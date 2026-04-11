source venv/bin/activate

# Full 100-epoch runs
python gan_mnist.py --config configs/vanilla.yaml
python gan_mnist.py --config configs/dcgan.yaml

# After training — FID score
python generate_samples.py --model dcgan --n 10000
python compute_fid.py --model dcgan

# Visualisations
python visualize.py --compare

source venv/bin/activate

# Full 100-epoch runs
python gan_mnist.py --config configs/vanilla.yaml
python gan_mnist.py --config configs/dcgan.yaml

# After training — FID score
python generate_samples.py --model dcgan --n 10000
python compute_fid.py --model dcgan

# Visualisations
python visualize.py --compare






----------------------------------------------------------------------------

# after this - python gan_mnist.py --config configs/dcgan.yaml

python gan_mnist.py --config configs/dcgan.yaml
# Generate samples for FID
python generate_samples.py --model vanilla --n 10000
python generate_samples.py --model dcgan --n 10000

# Compute FID for both
python compute_fid.py --model vanilla
python compute_fid.py --model dcgan

# Make all visualizations
python visualize.py --model vanilla
python visualize.py --model dcgan
python visualize.py --compare

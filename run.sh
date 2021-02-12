# Create necessary directories
mkdir log
mkdir output_dir

# Generate listops dataset
python listops.py

# Run Addition task with I-Bert
python AutoEncode.py --net ibert --seq_type fib --log true --exp 1

# For ListOps dataset, we run separate python code
python Classifier.py --net ibert --model_size small
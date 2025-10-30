# wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
import os
import subprocess

cifar10_path = 'datasets/cifar-10-python.tar.gz'
cifar100_path = 'datasets/cifar-100-python.tar.gz'

os.makedirs('datasets', exist_ok=True)

if not os.path.isfile(cifar10_path):
    print(f"{cifar10_path} not found. Downloading...")
    subprocess.run([
        'wget',
        '-O', cifar10_path,
        'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    ], check=True)
else:
    print(f"{cifar10_path} already exists.")

if not os.path.isfile(cifar100_path):
    print(f"{cifar100_path} not found. Downloading...")
    subprocess.run([
        'wget',
        '-O', cifar100_path,
        'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    ], check=True)
else:
    print(f"{cifar100_path} already exists.")

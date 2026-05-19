import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()

    # Nastavitev argumentov
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    
    args = parser.parse_args()

    # Sestavljanje ukaza
    command = [
        "python", 
        "STU-Net/nnUNet-2.2/nnunetv2/run/run_training.py",
        args.dataset_name_or_id,
        args.configuration,
        args.fold
    ]

    # Izvedba ukaza
    print(f"Zaganjam ukaz: {' '.join(command)}")
    subprocess.run(command)

if __name__ == "__main__":
    main()

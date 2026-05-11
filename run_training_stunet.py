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
    parser.add_argument('-tr', "--trainer", type=str, required=False, default='STUNetTrainer_base_ft',
                        help='specify a custom trainer')
    parser.add_argument('-pretrained_weights', type=str, required=False, default="code/base_ep4k.model",
                        help='stunet pretrained weights path')
    parser.add_argument('-p', '--plan_id', type=str, required=False, default='nnUNetPlans_custom',
                        help='specify a custom plan')

    args = parser.parse_args()

    # Sestavljanje ukaza
    command = [
        "python", 
        "STU-Net/nnUNet-2.2/nnunetv2/run/run_finetuning_stunet.py",
        args.dataset_name_or_id,
        args.configuration,
        args.fold,
        "-pretrained_weights", args.pretrained_weights,
        "-tr", args.trainer,
        "-p", args.plan_id
    ]

    # Izvedba ukaza
    print(f"Zaganjam ukaz: {' '.join(command)}")
    subprocess.run(command)

if __name__ == "__main__":
    main()

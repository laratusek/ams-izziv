import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Zagon inference.")

    # Definiranje argumentov na podlagi tvojega ukaza
    parser.add_argument("-i", "--input", help="Vhodna mapa s slikami")
    parser.add_argument("-o", "--output", help="Izhodna mapa za rezultate")
    parser.add_argument("-d", "--dataset", help="ID nabora podatkov (npr. 1)")
    parser.add_argument("-tr", "--trainer", default="nnUNetTrainer", help="Ime uporabljene trainer klase")
    parser.add_argument("-c", "--config", default="3d_fullres", help="Konfiguracija (npr. 3d_fullres)")
    parser.add_argument("-chk", "--checkpoint", default="checkpoint_best.pth", help="Ime checkpoint datoteke")
    parser.add_argument("-f", "--folds", default="0", help="Foldi, ki se uporabijo (npr. 0 ali 0 1 2)")
    parser.add_argument("-npp", "--num_proc_preprocessing", default="1", help="Število procesov za predpripravo")
    parser.add_argument("-nps", "--num_proc_segmentation", default="1", help="Število procesov za segmentacijo")

    args = parser.parse_args()

    # Sestavljanje ukaza
    # Uporabimo python -c za klic vstopne točke nnU-Net-a
    python_code = "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()"
    
    command = [
        "python", "-c", python_code,
        "-i", args.input,
        "-o", args.output,
        "-d", args.dataset,
        "-tr", args.trainer,
        "-c", args.config,
        "-chk", args.checkpoint,
        "-f", args.folds,
        "-npp", args.num_proc_preprocessing,
        "-nps", args.num_proc_segmentation
    ]

    # Izpis in zagon
    print(f"Zaganjam napovedovanje na naboru: {args.dataset}...")
    print(f"Ukaz: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        print("Napovedovanje uspešno zaključeno.")
    except subprocess.CalledProcessError as e:
        print(f"Prišlo je do napake pri izvajanju: {e}")

if __name__ == "__main__":
    main()

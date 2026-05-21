import shutil
import re
import os
import json

vhodna_mapa = "data"
izhodna_mapa = "data_test/Dataset001_ImageCAS"

imagesTs = os.path.join(izhodna_mapa, "imagesTs")
labelsTs = os.path.join(izhodna_mapa, "labelsTs")

for mapa in [imagesTs, labelsTs]:
    os.makedirs(mapa, exist_ok=True)

podmape = ["1-200", "201-400", "401-600", "601-800", "801-1000"]

for podmapa in podmape:
    pot_podmape = os.path.join(vhodna_mapa, podmapa)
    if not os.path.exists(pot_podmape):
        continue

    for datoteka in sorted(os.listdir(pot_podmape)):
        datoteka_path = os.path.join(pot_podmape, datoteka)
        
        if not (os.path.isfile(datoteka_path) and datoteka.endswith(".nii.gz")):
            continue

        # SLIKE
        if datoteka.endswith(".img.nii.gz"):
            match = re.match(r"(\d+)\.img\.nii\.gz", datoteka)
            if match:
                idx_str = match.group(1)
                idx = int(idx_str)

                # Omejitev: samo indeksi med 1 in 700
                if 751 <= idx <= 810:
                    novo_ime = f"{idx_str}_0000.nii.gz"
                    shutil.copy2(datoteka_path, os.path.join(imagesTs, novo_ime))

        # MASKE (Labels)
        elif datoteka.endswith(".label.nii.gz"):
            match = re.match(r"(\d+)\.label\.nii\.gz", datoteka)
            if match:
                idx_str = match.group(1)
                idx = int(idx_str)

                # Omejitev: samo indeksi med 751 in 810
                if 751 <= idx <= 810:
                    novo_ime = f"{idx_str}.nii.gz"
                    shutil.copy2(datoteka_path, os.path.join(labelsTs, novo_ime))

print("KONČANO: Slike in maske so kopirane na podlagi split_final.json.")

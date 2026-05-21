# AMS Izziv
Avtor: Lara Tušek Oklobdžija

Ta projekt vsebuje ukaze in skripte za fino učenje, inferenco in evalvacijo modelov STU-Net base in nnUNet na podatkovni zbirki ImageCAS.

Za (fino) učenje modelov je bila uporabljena razdelitev podatkov v splits_final_200.json (200 slik za učenje + 40 slik za validacijo). 

Koda in rezultati se nahajajo v mapi
`/media/FastDataMama/larat/ams_izziv/code`

Testni podatki in napovedi pa v mapi
`/media/FastDataMama/larat/ams_izziv/data_test`

## Pregled rezultatov

Modela sta bila testirana na 60 testnih primerih. Napovedi za oba modela se nahajajo znotraj mape .
Rezultati za klasične in topološke metrike so zbrani v tabeli.

| Model | Dice Score ↑ | IoU ↑ | Sens ↑ | clDice ↑ | VOI ↓ | $\beta$-Error ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **nnUNet** | 0,7326 | 0,5816 | 0,7657 | 0,7903 | 0,0143 | 16,2 |
| **STU-Net base** | 0,7702 | 0,6285 | 0,7329 | 0,8402 | 0,0116 | 10,9|

![Potek učenja STU-Net modela](img_metrics.png)

![Prikaz segmentacije v programu 3D slicer](img_slicer.png)



## Naučeni modeli

Naučeni modeli se nahajajo znotraj mape code/nnUNet_results_bck


---

## Zagon projekta

### Zahteve pred zagonom

```bash
docker build -t larat_amsizziv .
```

Pred izvajanjem ukazov se prepričajte, da ste pozicionirani v mapi:
`/media/FastDataMama/larat/ams_izziv/code`

### Fino učenje STU-Net

Zagon STU-Net treniranja preko pripravljene skripte. Skripta sprejme argumente: 

```bash
python code/run_training_stunet.py DATASET_NAME_OR_ID CONFIGURATION FOLD -tr TRAINER -pretrained_weights MODEL -p PLAN_ID
```

Skripto sem zagnala z naslednjimi argumenti:

```bash
docker run --rm -it --gpus 'device=0' --shm-size=16g \
-v "$(pwd)":/workdir/code \
-v /media/FastDataMama/new_nnunet/nnUNet/nnunet/nnUNet_preprocessed:/workdir/data/nnUNet_preprocessed \
-v /media/FastDataMama/larat/ams_izziv/code/splits_final_200.json:/workdir/data/nnUNet_preprocessed/Dataset001_ImageCAS/splits_final.json \
larat_amsizziv python code/run_training_stunet.py 
```

### Učenje nnUNet

Zagon nnUNet treniranja preko pripravljene skripte:
```bash
docker run --rm -it --gpus 'device=1' --shm-size=16g \
-v "$(pwd)":/workdir/code \
-v /media/FastDataMama/new_nnunet/nnUNet/nnunet/nnUNet_preprocessed:/workdir/data/nnUNet_preprocessed \
-v /media/FastDataMama/larat/ams_izziv/code/splits_final_200.json:/workdir/data/nnUNet_preprocessed/Dataset001_ImageCAS/splits_final.json \
-v /media/FastDataMama/Mark/U-mamba_AMS/data/nnUNet_raw:/workdir/data/nnUNet_raw \
larat_amsizziv run_training_nnunet.py
```

### Inferenca

Zagon inference preko pripravljene skripte. Skripta sprejme argumente.

```bash
python code/run_inference.py -i INPUT_PATH -o OUTPUT_PATH -d DATASET -tr TRAINER -c CONFIGURATION -chk CHECKPOINT -f FOLD -npp NUM_PROC_PREPROCESS -nps NUM_PROC_SEGMENTATION
```

Skripto sem zagnala na način:

```bash
docker run --rm -it --gpus 'device=0' --shm-size=16g \
-v "$(pwd)/code":/workdir/code \
-v "$(pwd)/data":/workdir/data \
larat_amsizziv python code/run_inference.py -i 'data/nnUNet_raw/Dataset001_ImageCAS/imagesTs' -o 'data/predictions/imagesTs_pred_stunet'
```

---

### Evalvacija

Primerjava napovedi (`nnUNet` vs `STU-Net`) z resničnimi podatki (`Ground Truth`) in izračun metrik v `.csv` datoteko:
Skripta sprejme argumente: 

```bash
python code/run_test.py -p1 PREDICTIONS_PATH1 -p2 PREDICTIONS_PATH2 -gt GT_PATH -o OUTPUT_PATH
```

Skripto sem zagnala na način:
```bash
docker run --rm -it --gpus 'device=0' --shm-size=16g \
-v "$(pwd)":/workdir/code \
-v /media/FastDataMama/larat/ams_izziv/data_test:/workdir/data_test \
larat_amsizziv python code/run_test.py -p1 data_test/predictions/imagesTs_pred_nnunet -p2 data_test/predictions/imagesTs_pred_stunet -gt data_test/Dataset001_ImageCAS/labelsTs -o code/results_metrics.csv

```
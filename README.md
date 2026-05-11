# ams-izziv

# docker poganjanje

```
docker built -t imedockerja .
docker run --rm -it --gpus all -v "$(pwd)":/workdir larat_amsizziv bash

docker run --rm -it --gpus all -v "$(pwd)":/workdir -v /media/FastDataMama/new_nnunet/nnUNet/nnunet/nnUNet_preprocessed:/workdir/data larat_amsizziv

```

# nalaganje utezi
```
curl -L -o "/media/FastDataMama/larat/ams-izziv/utezi/base_ep4k.model" "https://drive.usercontent.google.com/do
wnload?id=1BHCp1Ort-OaVFwaZmvsG4qHiKiPeNb4h&export=download&authuser=0"
```


/workdir/STU-Net/nnUNet-2.2/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py
from typing import Dict
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import variation_of_information
from skimage.morphology import skeletonize
import numpy as np
import cc3d
from scipy.ndimage import binary_fill_holes
import os
import csv
import argparse

def load_image(pot_do_datoteke: str, ciljni_tip=sitk.sitkUInt8) -> np.ndarray:
    """
    Naloži sliko in jo pretvori v numpy matriko.
    """

    slika = sitk.ReadImage(pot_do_datoteke)
    slika = sitk.Cast(slika, ciljni_tip)
    np_matrika = sitk.GetArrayFromImage(slika)

    # Binarizacija (foreground: vrednosti > 0)
    np_matrika = (np_matrika > 0).astype(np.uint8)
    return np_matrika


def clDice(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Izračun clDice metrike za 3D binarne maske
    """

    # topology precision
    tprec = np.sum(pred*skeletonize(gt))/np.sum(skeletonize(gt)) 

    # topology sensitivity
    tsens = np.sum(gt*skeletonize(pred))/np.sum(skeletonize(pred)) 

    # clDice metrika
    return 2*tprec*tsens/(tprec+tsens)


def voi(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Izračun Variation of Information (VOI) metrike za 3D binarne maske
    """
    
    # conditional entropies H(gt|pred), H(pred|gt)
    h_gt_pred, h_pred_gt = variation_of_information(gt, pred)

    # VOI metrika
    return h_gt_pred + h_pred_gt


def betti_num(maska: np.ndarray) -> list:
    """
    Izračuna [beta_0, beta_1, beta_2] za 3D binarno masko
    """
    # Zagotovimo, da je maska binarna (0 in 1)
    maska_bin = (maska > 0).astype(np.uint8)
    
    if np.sum(maska_bin) == 0:
        return [0, 0, 0]

    # Število ločenih objektov
    _, b0 = cc3d.connected_components(maska_bin, connectivity=26, return_N=True)

    # Število zaprtih votlin znotraj objektov
    zapolnjene_luknje = binary_fill_holes(maska_bin).astype(np.uint8)
    votline = zapolnjene_luknje - maska_bin
    
    # Število ločenih votlin v ozadju
    _, b2 = cc3d.connected_components(votline, connectivity=6, return_N=True)

    # Euler = b0 - b1 + b2  -->  b1 = b0 + b2 - Euler
    try:
        euler = cc3d.euler_characteristic(maska_bin, connectivity=26)
        b1 = b0 + b2 - euler
    except:
        b1 = 0 

    return [max(0, b0), max(0, b1), max(0, b2)]

def betti(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Izračun Betti error iz b0, b1, b2
    """
    gt_betti = betti_num(gt)
    pred_betti = betti_num(pred)

    b0_err = abs(gt_betti[0] - pred_betti[0])
    b1_err = abs(gt_betti[1] - pred_betti[1])
    b2_err = abs(gt_betti[2] - pred_betti[2])

    return float(b0_err + b1_err + b2_err)

def compute_metrics(gt: np.ndarray, pred: np.ndarray, pred_prob: np.ndarray | None = None) -> Dict[str, float]:
    """
    Vrne slovar metrik:
    {
    'dice': ..., 'iou': ..., 'sensitivity': ..., 'cldice': ..., 'voi': ..., 'betti': ...
    }
    """

    # Izračun TP, FP, TN, FN
    TP = np.sum((gt == 1) & (pred == 1))
    FP = np.sum((gt == 0) & (pred == 1))
    #TN = np.sum((gt == 0) & (pred == 0))
    FN = np.sum((gt == 1) & (pred == 0))    
    # Izračun metrik
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    if np.sum(gt) == 0 and np.sum(pred) == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0


    clDice_score  = clDice(pred, gt)
    voi_score = voi(pred, gt)
    betti_error = betti(pred, gt)

    metric_dict = {
    'dice': dice, 'iou': iou, 'sens': sens, 'cldice': clDice_score, 'voi': voi_score, 'betti': betti_error}   

    return metric_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Nastavitev argumentov
    parser.add_argument('-p1', '--predictions_path_1', type=str, required=True, help = 'Path to predictions of the first model')
    parser.add_argument('-p2', '--predictions_path_2', type=str, required=True, help = 'Path to predictions of the second model')
    parser.add_argument('-gt', '--gt_path', type=str, required=True, help = 'Path to ground truth masks')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output CSV file')

    args = parser.parse_args()
    
    # 1. Nastavitev poti do map
    output_csv = args.output_path

    vse_datoteke = [f for f in os.listdir(args.gt_path) if ".nii" in f]
    vse_datoteke.sort()

    print(f"Najdenih {len(vse_datoteke)} slik za evaluacijo.")

    # imena metrik v želenem vrstnem redu
    metrics_keys = ['dice', 'iou', 'sens', 'cldice', 'voi', 'betti']
    
    # Seznami za končni izračun povprečja
    rows = []

    # Zanka čez vse slike
    for ime_datoteke in vse_datoteke:
        id_slike = ime_datoteke.split(".")[0]  # ID slike brez končnice
        
        full_path_gt = os.path.join(args.gt_path, ime_datoteke)
        full_path_1 = os.path.join(args.predictions_path_1, ime_datoteke)
        full_path_2 = os.path.join(args.predictions_path_2, ime_datoteke)

        gt_mask = load_image(full_path_gt)
        pred_1 = load_image(full_path_1)
        pred_2 = load_image(full_path_2)

        # Izračun združenih metrik za oba modela
        metrics_1 = compute_metrics(gt_mask, pred_1)
        metrics_2 = compute_metrics(gt_mask, pred_2)

        # Izpis metrik
        print(f"Slika: {id_slike}")
        print(f"  Model 1: {metrics_1}")
        print(f"  Model 2: {metrics_2}")

        # Pretvorba slovarjev v urejena seznama števil
        vrednosti_1 = [metrics_1[k] for k in metrics_keys]
        vrednosti_2 = [metrics_2[k] for k in metrics_keys]      

        # Združitev obeh seznamov in zaokroževanje na 4 decimalke
        zaokrozene_vrednosti = [round(v, 4) for v in (vrednosti_1 + vrednosti_2)]
        rows.append([id_slike] + zaokrozene_vrednosti)

    # izračun končnega povprečja
    if rows:
        matrix_vals = np.array([r[1:] for r in rows])
        mean_vals = np.mean(matrix_vals, axis=0)
        rounded_means = [round(v, 4) for v in mean_vals]
        rows.append(["Average"] + rounded_means)

    # shranjevanje v CSV
    # priprava CSV glave
    headers = ["ID_Slike"] + [f"{m}_{k.title()}" for m in ["model1", "model2"] for k in metrics_keys]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)      
        writer.writerows(rows)  
                
    print(f"Končano! Rezultati so uspešno shranjeni v '{output_csv}'.")

import glob
from ase.io import read
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, csv
from tqdm import tqdm


ATR = {
    'H': 0.38, 'Li': 0.86, 'Be': 0.53, 'B': 1.01, 'C': 0.88, 'N': 0.86, 'O': 0.89, 'F': 0.82,
    'Na': 1.15, 'Mg': 1.28, 'Al': 1.53, 'Si': 1.38, 'P': 1.28, 'S': 1.20, 'Cl': 1.17, 'K': 1.44,
    'Ca': 1.17, 'Sc': 1.62, 'Ti': 1.65, 'V': 1.51, 'Cr': 1.53, 'Mn': 1.53, 'Fe': 1.43, 'Co': 1.31,
    'Ni': 1.33, 'Cu': 1.31, 'Zn': 1.41, 'Ga': 1.40, 'Ge': 1.35, 'As': 1.39, 'Se': 1.40, 'Br': 1.39,
    'Rb': 1.65, 'Sr': 1.30, 'Y': 1.84, 'Zr': 1.73, 'Nb': 1.66, 'Mo': 1.57, 'Ru': 1.58, 'Rh': 1.63,
    'Pd': 1.68, 'Ag': 1.56, 'Cd': 1.56, 'In': 1.53, 'Sn': 1.64, 'Sb': 1.64, 'Te': 1.65, 'I': 1.58,
    'Cs': 1.85, 'Ba': 1.52, 'La': 1.91, 'Ce': 1.98, 'Pr': 1.75, 'Nd': 1.92, 'Sm': 1.89, 'Eu': 1.83,
    'Gd': 1.79, 'Tb': 1.82, 'Dy': 1.79, 'Ho': 1.63, 'Er': 1.80, 'Tm': 1.84, 'Yb': 1.80, 'Lu': 1.86,
    'Hf': 1.73, 'W': 1.33, 'Re': 1.29, 'Ir': 1.50, 'Pt': 1.66, 'Au': 1.68, 'Hg': 1.88, 'Pb': 1.72,
    'Bi': 1.72, 'Th': 1.97, 'U': 1.76, 'Np': 1.73, 'Pu': 1.71
}

Coef_A = {
    'H': -0.6093, 'B': -2.2011, 'C': -1.2685, 'N': -1.2680, 'O': -1.0525, 'Cl': -0.7621, 'Br': -0.8003
}

Coef_C = {
    'H': 0.5927, 'B': 3.4380, 'C': 1.8855, 'N': 1.8401, 'O': 1.5189, 'Cl': 1.3723, 'Br': 1.5272
}

metals = ['Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Rb','Sr','Y','Zr','Nb','Mo','Ru','Rh',
         'Pd','Ag','Cd','In','Sn','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','W','Re','Re','Ir',
         'Pt','Au','Hg','Pb','Bi','Th','U','Np','Pu'] # do not include metalloids

def Chen_Manz(file):
        
    try:
        overlapping, under_carbon, over_carbon, isolated, misplaced_hydro = False, False, False, False, False
        atoms = read(file)
        sym = atoms.get_chemical_symbols()
        for a in range(len(atoms)):
            H_connected = []
            nl = []
            for b in range(len(atoms)):
                if a == b:
                    continue
                d = atoms.get_distance(a, b, mic=True)
                if sym[a] == 'H':
                    if d <= (0.3 + ATR.get(sym[a], 0.8) + ATR.get(sym[b], 0.8)):
                        H_connected.append(sym[b])
                if d < 0.5 * (ATR.get(sym[a], 0.8) + ATR.get(sym[b], 0.8)):
                    overlapping = True
                if d <= (ATR.get(sym[a], 0.8) + ATR.get(sym[b], 0.8)):
                    nl.append(b)
            if sym[a] == 'C':
                bonded_ele = [sym[e] for e in nl if sym[e] not in Coef_A]
                if not bonded_ele:
                    BO = []
                    for b in nl:
                        d = atoms.get_distance(a, b, mic=True)
                        BO_ab = 10 ** (Coef_A[sym[b]] * d + Coef_C[sym[b]])
                        if sym[b] == 'H' and BO_ab > 1.25:
                            BO_ab = 1.25
                        BO.append(BO_ab)
                    sum_BO = sum(BO)
                    if sum_BO < 3.3:
                        under_carbon = True
                    elif sum_BO >= 5.5:
                        over_carbon = True
            if len(nl) == 0:
                isolated = True
            if sym[a] == 'H':
                if ('N' in H_connected) or ('O' in H_connected):
                    common_metals = [e for e in H_connected if e in metals]
                    if len(common_metals) != 0:
                        misplaced_hydro = True
        return os.path.basename(file).replace(".cif", ""), overlapping, under_carbon, over_carbon, isolated, misplaced_hydro
    except Exception:
        return os.path.basename(file).replace(".cif", ""), "unknown", "unknown", "unknown", "unknown", "unknown"


def main():
    cif_files = glob.glob("/mnt/d/GithubDesk/FT-LLM-MOF/use/ToBaCCo/*.cif")

    if not cif_files:
        print("No CIF files found in ./all")
        return

    output_csv = "./Chen-Manz_ToBaCCo.csv"
    if os.path.exists(output_csv):
        os.remove(output_csv)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "overlapping", "under_carbon", "over_carbon", "isolated", "misplaced_hydro"])

    max_workers = 8

    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
         open(output_csv, mode="a", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)
        future_to_path = {
            executor.submit(Chen_Manz, path): path
            for path in cif_files
        }
        for future in tqdm(as_completed(future_to_path),
                           total=len(future_to_path),
                           desc="Processing CIFs"):
            try:
                result = future.result()
                print(result)
                writer.writerow(result)
                f_out.flush()
            except Exception as e:
                cif_path = future_to_path[future]
                print(f"Error processing {cif_path}: {e}")

if __name__ == "__main__":
    main()
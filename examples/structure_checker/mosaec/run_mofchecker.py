import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from mofchecker import MOFChecker
from tqdm import tqdm


def process_cif(cif_path):

    name = os.path.basename(cif_path).replace(".cif", "")
    try:
        mofchecker = MOFChecker.from_cif(cif_path)
        return (
            name,
            mofchecker.has_atomic_overlaps,
            mofchecker.has_overcoordinated_c,
            mofchecker.has_overcoordinated_n,
            mofchecker.has_overcoordinated_h,
            mofchecker.has_suspicicious_terminal_oxo,
            mofchecker.has_undercoordinated_c,
            mofchecker.has_undercoordinated_n,
            mofchecker.has_lone_molecule,
            mofchecker.has_high_charges,
            mofchecker.has_metal,
            mofchecker.has_carbon,
            mofchecker.is_porous
        )
    except Exception:

        return (
            name,
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown"
        )

def main():

    cif_folder = "./"
    cif_paths = glob.glob(os.path.join(cif_folder, "*cif"))

    if not cif_paths:
        print("No CIF")
        return


    remaining_cif_paths = [ path for path in cif_paths ]

    output_csv = "./MOFChecker.csv"
    header = [
        "name",
        "has_atomic_overlaps",
        "has_overcoordinated_c",
        "has_overcoordinated_n",
        "has_overcoordinated_h",
        "has_suspicicious_terminal_oxo",
        "has_undercoordinated_c",
        "has_undercoordinated_n",
        "has_lone_molecule",
        "has_high_charges",
        "has_metal",
        "has_carbon",
        "is_porous"
    ]

    if os.path.exists(output_csv):
        os.remove(output_csv)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    max_workers = 8

    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
         open(output_csv, mode="a", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)

        future_to_cif = {executor.submit(process_cif, path): path for path in remaining_cif_paths}

        for future in tqdm(as_completed(future_to_cif), total=len(future_to_cif), desc="Processing CIFs"):
            result = future.result()
            writer.writerow(result)
            f_out.flush()

if __name__ == "__main__":
    main()

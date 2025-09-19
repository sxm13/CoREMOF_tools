from CoREMOF.curate import mof_check
from CoREMOF.curate import run_mofclassifier
from CoREMOF.curate import run_MOSAEC
mof_check(structure="./examples/cal/IRMOF-1.cif", output_folder="./examples/cal/")
run_mofclassifier(cif_folder="./examples/checker", save_path="./mofclassifier_results.json", model="core", batch_size=64)
run_MOSAEC(cif_folder="./examples/checker", save_path="./", max_workers=64)
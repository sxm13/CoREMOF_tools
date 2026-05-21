from MOFClassifier import CLscore
import glob
import pandas as pd


all_structures = [stuc for stuc in glob.glob("/mnt/d/GithubDesk/FT-LLM-MOF/use/ToBaCCo/*cif")]
results = CLscore.predict_batch(root_cifs=all_structures, model="core", batch_size=16)

data = []
for i in range(len(all_structures)):
    data.append([results[i][0], results[i][2]])
pd.DataFrame(data, columns=["structure", "CLscore"]).to_csv("MOFClassifier_ToBaCCo.csv", index=False)
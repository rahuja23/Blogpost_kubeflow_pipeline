import json

import argparse
import os
def accuracy(args):
    sklearn_folder = args["sklearn_folder"]
    logistic_folder =  args["logistic_folder"]
    torch_folder = args["torch_folder"]
    svm_folder = args["svm_folder"]
    output_folder = args["output_dir"]
    with open(sklearn_folder) as f:
        sklearn_score= float(f.read())
    with open(logistic_folder) as f:
        logistic_score = float(f.read())
    with open(torch_folder) as f:
        torch_score = float(f.read())
    with open(svm_folder) as f:
        svm_score = float(f.read())
    metrics = {
        'metrics': [
            {
                'name': 'sklearn_score',
                'numberValue': sklearn_score,
                'format': "PERCENTAGE",
            },
            {
                'name': 'logistic_score',
                'numberValue': logistic_score,
                'format': "PERCENTAGE",
            },
            {
                'name': 'torch_score',
                'numberValue': torch_score,
                'format': "PERCENTAGE",
            },
            {
                'name': 'svm_score',
                'numberValue': svm_score,
                'format': "PERCENTAGE",
            },
        ]
    }
    json_object = json.dumps(metrics, indent=4)
    with open(os.path.join(output_folder, "metrics.json"), "w") as outfile:
        outfile.write(json_object)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sklearn_folder", type=str)
    parser.add_argument("--logistic_folder", type = str)
    parser.add_argument("--torch_folder", type= str)
    parser.add_argument("--svm_folder", type= str)
    args = parser.parse_args("--output_dir")
    return args
if __name__=="__main__":
    args = parse_args()
    accuracy(args)

import json
from typing import NamedTuple
import argparse
def accuracy(args) -> NamedTuple('Outputs', [('mlpipeline_metrics', 'Metrics')]):
    sklearn_score = args.sklearn_score
    logistic_score = args.logistic_score
    torch_score = args.torch_score
    svm_score = args.svm_score
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
    return [json.dumps(metrics)]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sklearn_score", type=float)
    parser.add_argument("--logistic_score", type = float)
    parser.add_argument("--torch_score", type= float)
    parser.add_argument("--svm_score", type= float)

    args = parser.parse_args()
    return args
if __name__=="__main__":
    args = parse_args()
    accuracy(args)

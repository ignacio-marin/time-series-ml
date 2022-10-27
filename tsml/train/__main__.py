import argparse

from tsml.load.data_loader import DataLoader
from tsml.train.trainer import TSTrainer
from tsml.train.bike_sharing.pipelines import xgb_pipeline_1, lgbm_pipeline_1
from tsml.train.helpers import run_experiment
from tsml.utils.config import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--client', 
    type=str, 
    required=True
)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.client == 'walmart':
        config = Config.from_json('config/walmart/train.json')
        dl = DataLoader(**config.data_loader)
        trainer = TSTrainer(config.trainer)
        pass

    if args.client == 'bike-sharing':
            
        config = Config.from_json('config/bike_sharing/train.json')
        dl = DataLoader(**config.data_loader)
        trainer = TSTrainer(config.trainer)
        for pipeline in [xgb_pipeline_1, lgbm_pipeline_1]:
            run_experiment(pipeline = pipeline, trainer = trainer, dl = dl, export=True, log= True)

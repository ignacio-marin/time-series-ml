
from sklearn.pipeline import Pipeline

from tsml.load.data_loader import DataLoader
from tsml.train.trainer import TSTrainer

def run_experiment(pipeline: Pipeline, trainer: TSTrainer, dl: DataLoader, exp_name:str = '',*,export=False,log=False):
    trainer.validate(pipeline, dl.X_train, dl.y_train)
    trainer.test(pipeline,  dl.X_train, dl.y_train, dl.X_test, dl.y_test)
    
    if export:
        pass

    if log:
        pass

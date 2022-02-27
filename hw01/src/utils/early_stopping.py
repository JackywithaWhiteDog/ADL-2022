from pytorch_lightning.callbacks import EarlyStopping

class EarlyStoppingWarmup(EarlyStopping):
    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.warmup:
            super()._run_early_stopping_check(trainer)

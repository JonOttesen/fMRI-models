{
    "name": "FastMRI_Unet",
    "n_gpu": 1,

    "arch": {
        "type": "Unet",
        "args": {}
    },
    "data_loader": {
        "type": "FastMRI",
        "args":{
            "data_dir": "data/",
            "batch_size": 128,
            "shuffle": True,
            "validation_split": 0.1,
            "num_workers": 2
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001,
            "weight_decay": 0,
            "amsgrad": True
        }
    },
    "loss": "nll_loss",
    "metrics": [
        "accuracy", "top_k_acc"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 100,

        "save_dir": ".saved/",
        "save_period": 1,
        "verbosity": 2,

        "monitor": "min val_loss",
        "early_stop": 10,

    }
}

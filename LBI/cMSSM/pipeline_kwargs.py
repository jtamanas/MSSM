pipeline_kwargs = {
    # Model hyperparameters
    "model_type": "classifier",  # "classifier" or "flow"
    "ensemble_size": 5,
    "num_layers": 7,
    "hidden_dim": 128,
    # Optimizer hyperparmeters
    "max_norm": 1e-3,
    "learning_rate": 3e-4,
    "weight_decay": 1e-1,
    # Train hyperparameters
    "nsteps": 250000,
    "patience": 15,
    "eval_interval": 100,
    # Dataloader hyperparameters
    "batch_size": 128,
    "train_split": 0.9,
    "num_workers": 0,
    "add_noise": True,
    # Sequential hyperparameters
    "num_rounds": 3,
    "num_initial_samples": 1500,
    "num_samples_per_round": 100 // 10,
    "num_warmup_per_round": 1000,
    "num_chains": 10,
    "logger": None,
}

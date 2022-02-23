pipeline_kwargs = {
    # Model hyperparameters
    "model_type": "classifier",  # "classifier" or "flow"
    "ensemble_size": 5,
    "num_layers": 5,
    "hidden_dim": 256,
    # Optimizer hyperparmeters
    "max_norm": 1e-3,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    # Train hyperparameters
    "nsteps": 250000,
    "patience": 20,
    "eval_interval": 100,
    # Dataloader hyperparameters
    "batch_size": 128,
    "train_split": 0.9,
    "num_workers": 0,
    "add_noise": True,
    # Sequential hyperparameters
    "num_rounds": 1,
    "num_initial_samples": 50000,
    "num_samples_per_round": 10000 // 20,
    "num_warmup_per_round": 5000,
    "num_chains": 20,
    "logger": None,
}

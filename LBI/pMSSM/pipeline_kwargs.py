pipeline_kwargs = {
    # simulator kwargs
    "simulator_kwargs": {"use_direct_detection": False, "use_atlas_constraints": True},
    # Model hyperparameters
    "model_type": "classifier",  # "classifier" or "flow"
    "ensemble_size": 5,
    "num_layers": 5,
    "hidden_dim": 512,
    # flow specific parameters
    "transform_type": "MaskedPiecewiseRationalQuadraticAutoregressiveTransform",
    "permutation": "Conv1x1",
    "tail_bound": 10.0,
    "num_bins": 4,
    # Optimizer hyperparmeters
    "max_norm": 1e-3,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    # Train hyperparameters
    "nsteps": 250000,
    "patience": 30,
    "eval_interval": 100,
    # Dataloader hyperparameters
    "batch_size": 128,
    "train_split": 0.9,
    "num_workers": 0,
    "add_noise": True,
    # Sequential hyperparameters
    "num_rounds": 1,
    "num_initial_samples": 10,
    "num_samples_per_round": 1000 // 10,
    "num_warmup_per_round": 2000,
    "num_chains": 1,
    "logger": None,
}

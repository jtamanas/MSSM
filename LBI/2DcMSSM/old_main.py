import jax
import jax.numpy as np
import numpy as onp
import optax
from trax.jaxboard import SummaryWriter
from lbi.prior import SmoothedBoxPrior
from lbi.dataset import getDataLoaderBuilder

# from lbi.diagnostics import MMD, ROC_AUC, LR_ROC_AUC
from lbi.sequential.sequential import sequential
from lbi.models import parallel_init_fn
from lbi.models.steps import get_train_step, get_valid_step
from lbi.models.flows import construct_MAF
from lbi.models.MLP import MLP
from lbi.models.classifier import construct_Classifier
from lbi.models.classifier.classifier import get_loss_fn
from lbi.trainer import getTrainer
from lbi.sampler import hmc
from simulator import get_simulator

import corner
import matplotlib.pyplot as plt
import datetime


@jax.jit
def process_X(x):
    omega = (np.log(x[:, 0]) - 1.6271843) / 2.257083
    gmuon = (1e10 * x[:, 1] - 0.686347) / (3.0785 * 3)
    mh = (x[:, 2] - 124.86377) / 2.2839
    out = np.stack([omega, gmuon, mh], axis=1)
    return out


# --------------------------
model_type = "flow"  # "classifier" or "flow"

seed = 1234
rng, model_rng, hmc_rng = jax.random.split(jax.random.PRNGKey(seed), num=3)

# Model hyperparameters
ensemble_size = 15
num_layers = 2
hidden_dim = 32

# Optimizer hyperparmeters
max_norm = 1e-3
learning_rate = 3e-4
weight_decay = 1e-1
sync_period = 5
slow_step_size = 0.5

# Train hyperparameters
nsteps = 50000
patience = 1500
eval_interval = 10  # an epoch ~ 750 steps

# Sequential hyperparameters
num_rounds = 1
num_chains = 10
num_initial_samples = 1500
num_samples_per_round = 1000 // num_chains
num_warmup_per_round = 1000

# --------------------------
# Create logger

experiment_name = datetime.datetime.now().strftime("%s")
experiment_name = f"{model_type}_{experiment_name}"
logger = SummaryWriter("runs/" + experiment_name)
logger = None


# --------------------------
# set up simulation and observables
simulate, obs_dim, theta_dim = get_simulator(preprocess=process_X)

# set up true model for posterior inference test
X_true = np.array([[0.12, 251e-11, 125.0]])
X_true = process_X(X_true)
print("X_true", X_true)

data_loader_builder = getDataLoaderBuilder(
    sequential_mode=model_type,
    batch_size=128,
    train_split=0.95,
    num_workers=0,
    add_noise=False,
)

# --------------------------
# set up prior
log_prior, sample_prior = SmoothedBoxPrior(
    theta_dim=theta_dim, lower=0, upper=1, sigma=0.01
)

# TODO: Package model, optimizer, trainer initialization into a function

# --------------------------
# Create optimizer
optimizer = optax.chain(
    # Set the parameters of Adam optimizer
    optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    ),
    optax.adaptive_grad_clip(max_norm),
)

# --------------------------
# Create model
if model_type == "classifier":
    classifier_kwargs = {
        # "output_dim": 1,
        "hidden_dim": (obs_dim + theta_dim) * 2,
        "num_layers": 2,
        "use_residual": False,
        "act": "leaky_relu",
    }
    model, loss_fn = construct_Classifier(**classifier_kwargs)
else:
    maf_kwargs = {
        "rng": rng,
        "input_dim": obs_dim,
        "hidden_dim": hidden_dim,
        "context_dim": theta_dim,
        "n_layers": num_layers,
        "permutation": "Conv1x1",
        "normalization": None,
        "made_activation": "gelu",
    }
    context_embedding_kwargs = {
        "output_dim": theta_dim * 2,
        "hidden_dim": theta_dim * 2,
        "num_layers": 2,
        "act": "leaky_relu",
    }

    context_embedding = MLP(**context_embedding_kwargs)
    model, loss_fn = construct_MAF(context_embedding=context_embedding, **maf_kwargs)

params, opt_state = parallel_init_fn(
    jax.random.split(rng, ensemble_size),
    model,
    optimizer,
    (obs_dim,),
    (theta_dim,),
)

# the models' __call__ are their log_prob fns
parallel_log_prob = jax.vmap(model.apply, in_axes=(0, None, None))

# --------------------------
# Create trainer

train_step = get_train_step(loss_fn, optimizer)
valid_step = get_valid_step({"valid_loss": loss_fn})

trainer = getTrainer(
    train_step,
    valid_step=valid_step,
    nsteps=nsteps,
    eval_interval=eval_interval,
    patience=patience,
    logger=logger,
    train_kwargs=None,
    valid_kwargs=None,
)

# Train model sequentially
params, Theta_post = sequential(
    rng,
    X_true,
    params,
    parallel_log_prob,
    log_prior,
    sample_prior,
    simulate,
    opt_state,
    trainer,
    data_loader_builder,
    num_rounds=num_rounds,
    num_initial_samples=num_initial_samples,
    num_warmup_per_round=num_warmup_per_round,
    num_samples_per_round=num_samples_per_round,
    num_samples_per_theta=1,
    num_chains=num_chains,
    logger=logger,
)


def potential_fn(theta):
    if len(theta.shape) == 1:
        theta = theta[None, :]

    log_L = parallel_log_prob(params, X_true, theta)
    log_L = log_L.mean(axis=0)

    log_post = -log_L - log_prior(theta)
    return log_post.sum()


num_chains = 16
init_theta = sample_prior(rng, num_samples=num_chains)

corner.corner(
    onp.array(sample_prior(rng, num_samples=10000)),
    range=[(0, 1) for i in range(theta_dim)],
    bins=75,
    smooth=(1.0),
    smooth1d=(1.0),
)


if hasattr(logger, "plot"):
    logger.plot(f"Prior Corner Plot", plt, close_plot=True)
else:
    plt.savefig("prior_corner.png")

mcmc = hmc(
    rng,
    potential_fn,
    init_theta,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=True,
    step_size=1e0,
    max_tree_depth=8,
    num_warmup=2000,
    num_samples=500,
    num_chains=num_chains,
    extra_fields=("potential_energy",),
    chain_method="vectorized",
)
mcmc.print_summary()


samples = mcmc.get_samples(group_by_chain=False).squeeze()
posterior_log_probs = -mcmc.get_extra_fields()["potential_energy"]
samples = np.hstack([samples, posterior_log_probs[:, None]])


labels = list(map(r"$\theta_{{{0}}}$".format, range(1, theta_dim + 1)))
labels = [r"$M_0$", r"$M_{1/2}$"]
labels += ["log prob"]
ranges = [
    (0.0, 1.0),
    (0.0, 1.0),
    (posterior_log_probs.min(), posterior_log_probs.max()),
]

corner.corner(onp.array(samples), range=ranges, labels=labels)

if hasattr(logger, "plot"):
    logger.plot(f"Posterior Corner Plot", plt, close_plot=True)
else:
    plt.savefig("flow_posterior_corner.png")

# data = simulate(rng, theta_samples, num_samples_per_theta=1)
#
# if model_type == "classifier":
#     fpr, tpr, auc = LR_ROC_AUC(
#         rng,
#         model_params,
#         log_pdf,
#         data,
#         theta_samples,
#         data_split=0.05,
#     )
# else:
#     model_samples = sample(rng, model_params, theta_samples)
#     fpr, tpr, auc = ROC_AUC(
#         rng,
#         data,
#         model_samples,
#     )

# # Optimal discriminator
# plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc)
# plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), linestyle="--", color="black")
# plt.legend(loc="lower right")

# if hasattr(logger, "plot"):
#     logger.plot(f"ROC", plt, close_plot=True)
# else:
#     plt.show()


if hasattr(logger, "close"):
    logger.close()

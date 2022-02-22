import jax.numpy as np
import numpy as onp
from simulator import get_simulator, theta_addunits
from pymicromegas import MicromegasSettings


def seq_vs_parallel_test(num_tests=100, nan_to_num=394786538756):
  sim, obs_dim, theta_dim = get_simulator(micromegas_simulator="spheno")
  test_theta = onp.random.rand(num_tests, theta_dim)

  seq_results = []
  # sequential
  for i in range(num_tests):
    seq_results.append(sim(None, test_theta[i : i + 1]))

  parallel_results = []

  parallel_results.append(sim(None, test_theta))
  seq_results = onp.array(seq_results).squeeze()
  seq_results = onp.nan_to_num(seq_results, nan_to_num)
  parallel_results = onp.array(parallel_results).squeeze()
  parallel_results = onp.nan_to_num(parallel_results, nan_to_num)

  assert onp.allclose(seq_results, parallel_results)
  print("===========================")
  print("Passed sequential vs parallel test")
  print("===========================")



def check_theta_add_units():
  sim, obs_dim, theta_dim = get_simulator(micromegas_simulator="spheno")
  test_theta = onp.ones((1, theta_dim))
  print(theta_addunits(test_theta))
  print(theta_addunits(-test_theta))


if __name__ == "__main__":
    # seq_vs_parallel_test()
    check_theta_add_units()
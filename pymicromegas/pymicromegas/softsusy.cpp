#include "../../include/micromegas.h"
#include "../../include/micromegas_aux.h"
#include "../lib/pmodel.h"
#include "pymicromegas.hpp"
#include <bitset>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static const MicromegasSettings default_settings{};

void SugraParameters::initialize() const {
  double gMG1 = p_mhf;
  double gMG2 = p_mhf;
  double gMG3 = p_mhf;
  double gAl = p_a0;
  double gAt = p_a0;
  double gAb = p_a0;
  double gMHu = p_m0;
  double gMHd = p_m0;
  double gMl2 = p_m0;
  double gMl3 = p_m0;
  double gMr2 = p_m0;
  double gMr3 = p_m0;
  double gMq2 = p_m0;
  double gMq3 = p_m0;
  double gMu2 = p_m0;
  double gMd2 = p_m0;
  double gMu3 = p_m0;
  double gMd3 = p_m0;

  int err =
      softSusySUGRA(p_tb, gMG1, gMG2, gMG3, gAl, gAt, gAb, p_sgn, gMHu, gMHd,
                    gMl2, gMl3, gMr2, gMr3, gMq2, gMq3, gMu2, gMu3, gMd2, gMd3);
  error_parser("softSusySUGRA", err);
}

void EwsbParameters::initialize() const {
  int err = softSusyEwsbMSSM();
  error_parser("softSusyEwsbMSSM", err);
}

void MicromegasResults::execute(const MicromegasSettings &settings,
                                const SugraParameters &params) {
  if (PyErr_CheckSignals() != 0) {
    throw py::error_already_set();
  }
  auto size = p_omega.size();
  try {
    params.initialize();
    execute(settings);
  } catch (const std::exception &e) {
    set_nans(size);
    py::print(e.what());
  }
}

void MicromegasResults::execute(const MicromegasSettings &settings,
                                const EwsbParameters &params) {
  if (PyErr_CheckSignals() != 0) {
    throw py::error_already_set();
  }
  auto size = p_omega.size();
  try {
    params.initialize();
    execute(settings);
  } catch (const py::error_already_set &e) {
    throw py::error_already_set();
  } catch (const std::exception &e) {
    set_nans(size);
    py::print(e.what());
  }
}

// ============================================================================
// ---- Python Interface ------------------------------------------------------
// ============================================================================

const std::string BASE_DOC_STRING_SUGRA = R"pbdoc(
Run micromegas with the parameters define at the GUT scale and settings using SoftSusy as an RGE backend.

Parameters
----------
params: SugraParameters
  Parameter object containing the model parameters defined at the GUT scale.
settings: MicromegasSettings, optional
  Settings object containing parameters to run micromegas with. Default is
  MicromegasSettings();

Returns
-------
results: MicromegasResults
  Results object containing requested results.
)pbdoc";

const std::string BASE_DOC_STRING_EWSB = R"pbdoc(
Run micromegas with the parameters define at the EW scale and settings using SoftSusy as an RGE backend.

Parameters
----------
params: SugraParameters
  Parameter object containing the model parameters defined at the EW scale.
settings: MicromegasSettings, optional
  Settings object containing parameters to run micromegas with. Default is
  MicromegasSettings();

Returns
-------
results: MicromegasResults
  Results object containing requested results.
)pbdoc";

const std::string DOC_STRING_VEC = R"pbdoc(
Same as `sugra(params, settings)` but with a vector of parameters.
)pbdoc";

PYBIND11_MODULE(softsusy, m) {
  py::module_::import("pymicromegas");
  m.doc() = "Python interface to micromegas with SoftSusy as an RGE backend.";
  // ================================================
  // ---- Single Parameter, SUGRA, with settings ----
  // ================================================
  m.def(
      "softsusy",
      [](const SugraParameters &params,
         const MicromegasSettings &settings = default_settings) {
        MicromegasResults results(1);
        results.execute(settings, params);
        return results;
      },
      BASE_DOC_STRING_SUGRA.c_str(), py::arg("params"),
      py::arg("settings") = default_settings, py::return_value_policy::move);
  // ====================================================
  // ---- Vector of Parameters, SUGRA, with settings ----
  // ====================================================
  m.def(
      "softsusy",
      [](const std::vector<SugraParameters> &params,
         const MicromegasSettings &settings = default_settings) {
        size_t batchsize = params.size();
        MicromegasResults output(batchsize);
        for (size_t i = 0; i < batchsize; i++) {
          output.execute(settings, params[i]);
        }
        return output;
      },
      DOC_STRING_VEC.c_str(), py::arg("params"),
      py::arg("settings") = default_settings, py::return_value_policy::move);
  // ===============================================
  // ---- Single Parameter, EWSB, with settings ----
  // ===============================================
  m.def(
      "softsusy",
      [](const EwsbParameters &params,
         const MicromegasSettings &settings = default_settings) {
        MicromegasResults results(1);
        results.execute(settings, params);
        return results;
      },
      BASE_DOC_STRING_EWSB.c_str(), py::arg("params"),
      py::arg("settings") = default_settings, py::return_value_policy::move);
  // ===================================================
  // ---- Vector of Parameters, EWSB, with settings ----
  // ===================================================
  m.def(
      "softsusy",
      [](const std::vector<EwsbParameters> &params,
         const MicromegasSettings &settings = default_settings) {
        size_t batchsize = params.size();
        MicromegasResults output(batchsize);
        for (size_t i = 0; i < batchsize; i++) {
          output.execute(settings, params[i]);
        }
        return output;
      },
      DOC_STRING_VEC.c_str(), py::arg("params"),
      py::arg("settings") = default_settings, py::return_value_policy::move);
}

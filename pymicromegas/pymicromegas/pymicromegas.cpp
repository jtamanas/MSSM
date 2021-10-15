#include "pymicromegas.hpp"
#include "../../include/micromegas.h"
#include "../../include/micromegas_aux.h"
#include "../lib/pmodel.h"
#include <bitset>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace py = pybind11;

static const std::array<std::string, 31> MASS_NAMES{
    "MSne", // Electron-snuetrino
    "MSnm", // Muon-Snuetrino
    "MSeL", // Left-handed Selectron
    "MSeR", // Right-handed Selectron
    "MSmL", // Left-handed Smuon
    "MSmR", // Right-handed Smuon
    "MSdL", // Left-handed down squark
    "MSdR", // Right-handed up squark
    "MSuL", // Left-handed up squark
    "MSuR", // Right-handed up squark
    "MSsL", // Left-handed strange squark
    "MSsR", // Right-handed strange squark
    "MScL", // Left-handed charm squark
    "MScR", // Right-handed charm squark
    "MSnl", // Left-handed tau snuetrino
    "MSl1", // Stau 1
    "MSl2", // Stau 2
    "MSb1", // Bottom squark 1
    "MSb2", // Bottom squark 2
    "MSt1", // Top squark 1
    "MSt2", // Top squark 2
    "MSG",  // Gluino mass
    "MNE1", // Neutralino 1
    "MNE2", // Neutralino 2
    "MNE3", // Neutralino 3
    "MNE4", // Neutralino 4
    "MC1",  // Chargino 1
    "MC2",  // Chargino 2
    "Mh",   // Higgs mass
    "MH",   // Other higgs mass
    "MHc"   // Charged Higgs mass
};

static const std::array<int, 31> PDG_CODES{
    1000012, // Electron-snuetrino
    1000014, // Muon-Snuetrino
    1000011, // Left-handed Selectron
    2000011, // Right-handed Selectron
    1000013, // Left-handed Smuon
    2000013, // Right-handed Smuon
    1000001, // Left-handed down squark
    2000001, // Right-handed up squark
    1000002, // Left-handed up squark
    2000002, // Right-handed up squark
    1000003, // Left-handed strange squark
    2000003, // Right-handed strange squark
    1000004, // Left-handed charm squark
    2000004, // Right-handed charm squark
    1000016, // Left-handed tau snuetrino
    1000015, // Stau 1
    2000015, // Stau 2
    1000005, // Bottom squark 1
    2000005, // Bottom squark 2
    1000006, // Top squark 1
    2000006, // Top squark 2
    1000021, // Gluino mass
    1000022, // Neutralino 1
    1000023, // Neutralino 2
    1000025, // Neutralino 3
    1000035, // Neutralino 4
    1000024, // Chargino 1
    1000037, // Chargino 2
    25,      // Higgs mass
    35,      // Other higgs mass
    37       // Charged Higgs mass
};

void error_parser(const std::string &name, int err) {
  if (err == 0) {
    return;
  }
  if (err == -1) {
    throw std::runtime_error(name + " errored with -1: Canot open the file.\n");
  }
  if (err == -2) {
    throw std::runtime_error(
        name + " errored with -2: Error in spectrum calculator.\n");
  }
  if (err == -3) {
    throw std::runtime_error(name + " errored with -3: No data.\n");
  }
  if (err > 0) {
    throw std::runtime_error(name +
                             " errored with: Wrong file contents at line " +
                             std::to_string(err) + "\n");
  }
}

MicromegasResults::MicromegasResults(size_t n) {
  p_omega.reserve(n);
  p_xf.reserve(n);
  p_bsgsm.reserve(n);
  p_bsgnlo.reserve(n);
  p_deltarho.reserve(n);
  p_bsmumu.reserve(n);
  p_btaunu.reserve(n);
  p_gmuon.reserve(n);
  for (auto &arr : p_masses) {
    arr.reserve(n);
  }
}

void MicromegasResults::set_nans(size_t size) {
  if (p_omega.size() != size) {
    p_omega.back() = std::nan("0");
  } else {
    p_omega.push_back(std::nan("0"));
  }
  if (p_xf.size() != size) {
    p_xf.back() = std::nan("0");
  } else {
    p_xf.push_back(std::nan("0"));
  }
  if (p_bsgsm.size() != size) {
    p_bsgsm.back() = std::nan("0");
  } else {
    p_bsgsm.push_back(std::nan("0"));
  }
  if (p_bsgnlo.size() != size) {
    p_bsgnlo.back() = std::nan("0");
  } else {
    p_bsgnlo.push_back(std::nan("0"));
  }
  if (p_deltarho.size() != size) {
    p_deltarho.back() = std::nan("0");
  } else {
    p_deltarho.push_back(std::nan("0"));
  }
  if (p_bsmumu.size() != size) {
    p_bsmumu.back() = std::nan("0");
  } else {
    p_bsmumu.push_back(std::nan("0"));
  }
  if (p_btaunu.size() != size) {
    p_btaunu.back() = std::nan("0");
  } else {
    p_btaunu.push_back(std::nan("0"));
  }
  if (p_gmuon.size() != size) {
    p_gmuon.back() = std::nan("0");
  } else {
    p_gmuon.push_back(std::nan("0"));
  }
  for (size_t i = 0; i < p_masses.size(); i++) {
    if (p_masses[i].size() != size) {
      p_masses[i].back() = std::nan("0");
    } else {
      p_masses[i].push_back(std::nan("0"));
    }
  }
}

void MicromegasResults::execute(const MicromegasSettings &settings) {
  sort_odd_particles(settings);
  compute_relic_density(settings);
  compute_masses(settings);
  compute_bsg(settings);
  compute_deltarho(settings);
  compute_bsmumu(settings);
  compute_btaunu(settings);
  compute_gmuon(settings);
}

void MicromegasResults::sort_odd_particles(
    const MicromegasSettings &settings) const {
  char cdmName[10];
  int err = 0;
  if (settings.sort_odd_particles()) {
    err = sortOddParticles(cdmName);
    if (err != 0) {
      throw std::runtime_error(
          "sortOddParticles() errored with: Can't calculate " +
          std::string(cdmName) + "\n");
    }
  }
}

void MicromegasResults::compute_relic_density(
    const MicromegasSettings &settings) {
  if (settings.compute_rd()) {
    int err = 0;
    // to exclude processes with virtual W/Z in DM   annihilation
    VZdecay = 0;
    VWdecay = 0;
    cleanDecayTable();
    double xf = 0;
    double omega = darkOmega(&xf, settings.fast(), settings.beps(), &err);
    VZdecay = 1;
    VWdecay = 1;
    cleanDecayTable();
    p_omega.push_back(omega);
    p_xf.push_back(xf);
  } else {
    p_omega.push_back(std::nan("0"));
    p_xf.push_back(std::nan("0"));
  }
}

void MicromegasResults::compute_masses(const MicromegasSettings &settings) {
  if (settings.compute_masses()) {
    // Fill in all the masses
    for (size_t i = 0; i < PDG_CODES.size(); i++) {
      auto *name = pdg2name(PDG_CODES[i]);
      if (name != nullptr) {
        p_masses[i].push_back(pMass(name));
      }
    }
  } else {
    for (size_t i = 0; i < MASS_NAMES.size(); i++) {
      p_masses[i].push_back(std::nan("0"));
    }
  }
}

void MicromegasResults::compute_bsg(const MicromegasSettings &settings) {
  if (settings.compute_bsg()) {
    double bsgsm = 0;
    p_bsgnlo.push_back(bsgnlo(&bsgsm));
    p_bsgsm.push_back(bsgsm);
  } else {
    p_bsgnlo.push_back(std::nan("0"));
    p_bsgsm.push_back(std::nan("0"));
  }
}

void MicromegasResults::compute_deltarho(const MicromegasSettings &settings) {
  if (settings.compute_deltarho()) {
    p_deltarho.push_back(deltarho());
  } else {
    p_deltarho.push_back(std::nan("0"));
  }
}

void MicromegasResults::compute_bsmumu(const MicromegasSettings &settings) {
  if (settings.compute_bsmumu()) {
    p_bsmumu.push_back(bsmumu());
  } else {
    p_bsmumu.push_back(std::nan("0"));
  }
}

void MicromegasResults::compute_btaunu(const MicromegasSettings &settings) {
  if (settings.compute_btaunu()) {
    p_btaunu.push_back(btaunu());
  } else {
    p_btaunu.push_back(std::nan("0"));
  }
}

void MicromegasResults::compute_gmuon(const MicromegasSettings &settings) {
  if (settings.compute_gmuon()) {
    p_gmuon.push_back(gmuon());
  } else {
    p_gmuon.push_back(std::nan("0"));
  }
}

// ============================================================================
// ---- Python Interface ------------------------------------------------------
// ============================================================================

PYBIND11_MODULE(pymicromegas, m) {
  m.doc() = "Python interface to micromegas";

  py::class_<MicromegasSettings>(m, "MicromegasSettings")
      .def(py::init<bool, bool, bool, bool, bool, bool, bool, bool, bool,
                    double, double>(),
           "Create a settings object for use in running micrOmegas.",
           py::arg("relic_density") = true, py::arg("masses") = true,
           py::arg("gmuon") = true, py::arg("bsg") = true,
           py::arg("bsmumu") = true, py::arg("btaunu") = true,
           py::arg("delta_rho") = true, py::arg("sort_odd") = true,
           py::arg("fast") = true, py::arg("beps") = 1e-4,
           py::arg("cut") = 1e-2)
      .def_property(
          "fast", [](const MicromegasSettings &s) { return s.fast(); },
          [](MicromegasSettings &s) { return s.fast(); })
      .def_property(
          "beps", [](const MicromegasSettings &s) { return s.beps(); },
          [](MicromegasSettings &s) { return s.beps(); })
      .def_property(
          "cut", [](const MicromegasSettings &s) { return s.cut(); },
          [](MicromegasSettings &s) { return s.cut(); })
      .def_property(
          "compute_rd",
          [](const MicromegasSettings &s) { return s.compute_rd(); },
          [](MicromegasSettings &s) { return s.compute_rd(); })
      .def_property(
          "compute_masses",
          [](const MicromegasSettings &s) { return s.compute_masses(); },
          [](MicromegasSettings &s) { return s.compute_masses(); })
      .def_property(
          "compute_bsg",
          [](const MicromegasSettings &s) { return s.compute_bsg(); },
          [](MicromegasSettings &s) { return s.compute_bsg(); })
      .def_property(
          "compute_bsmumu",
          [](const MicromegasSettings &s) { return s.compute_bsmumu(); },
          [](MicromegasSettings &s) { return s.compute_bsmumu(); })
      .def_property(
          "compute_btaunu",
          [](const MicromegasSettings &s) { return s.compute_btaunu(); },
          [](MicromegasSettings &s) { return s.compute_btaunu(); })
      .def_property(
          "compute_deltarho",
          [](const MicromegasSettings &s) { return s.compute_deltarho(); },
          [](MicromegasSettings &s) { return s.compute_deltarho(); })
      .def_property(
          "compute_gmuon",
          [](const MicromegasSettings &s) { return s.compute_gmuon(); },
          [](MicromegasSettings &s) { return s.compute_gmuon(); })
      .def_property(
          "sort_odd_particles",
          [](const MicromegasSettings &s) { return s.sort_odd_particles(); },
          [](MicromegasSettings &s) { return s.sort_odd_particles(); });

  py::class_<MicromegasResults>(m, "MicromegasResults")
      .def("omega", [](const MicromegasResults &r) { return r.omega(); })
      .def("xf", [](const MicromegasResults &r) { return r.xf(); })
      .def("bsgsm", [](const MicromegasResults &r) { return r.b_sg_sm(); })
      .def("bsgnlo", [](const MicromegasResults &r) { return r.b_sg_nlo(); })
      .def("deltarho", [](const MicromegasResults &r) { return r.delta_rho(); })
      .def("bsmumu", [](const MicromegasResults &r) { return r.b_smumu(); })
      .def("btaunu", [](const MicromegasResults &r) { return r.b_taunu(); })
      .def("gmuon", [](const MicromegasResults &r) { return r.g_muon(); })
      .def("msne", [](const MicromegasResults &r) { return r.msne(); })
      .def("msnm", [](const MicromegasResults &r) { return r.msnm(); })
      .def("msel", [](const MicromegasResults &r) { return r.msel(); })
      .def("mser", [](const MicromegasResults &r) { return r.mser(); })
      .def("msml", [](const MicromegasResults &r) { return r.msml(); })
      .def("msmr", [](const MicromegasResults &r) { return r.msmr(); })
      .def("msdl", [](const MicromegasResults &r) { return r.msdl(); })
      .def("msdr", [](const MicromegasResults &r) { return r.msdr(); })
      .def("msul", [](const MicromegasResults &r) { return r.msul(); })
      .def("msur", [](const MicromegasResults &r) { return r.msur(); })
      .def("mssl", [](const MicromegasResults &r) { return r.mssl(); })
      .def("mssr", [](const MicromegasResults &r) { return r.mssr(); })
      .def("mscl", [](const MicromegasResults &r) { return r.mscl(); })
      .def("mscr", [](const MicromegasResults &r) { return r.mscr(); })
      .def("msnl", [](const MicromegasResults &r) { return r.msnl(); })
      .def("msl1", [](const MicromegasResults &r) { return r.msl1(); })
      .def("msl2", [](const MicromegasResults &r) { return r.msl2(); })
      .def("msb1", [](const MicromegasResults &r) { return r.msb1(); })
      .def("msb2", [](const MicromegasResults &r) { return r.msb2(); })
      .def("mst1", [](const MicromegasResults &r) { return r.mst1(); })
      .def("mst2", [](const MicromegasResults &r) { return r.mst2(); })
      .def("mg", [](const MicromegasResults &r) { return r.mg(); })
      .def("mneut1", [](const MicromegasResults &r) { return r.mneut1(); })
      .def("mneut2", [](const MicromegasResults &r) { return r.mneut2(); })
      .def("mneut3", [](const MicromegasResults &r) { return r.mneut3(); })
      .def("mneut4", [](const MicromegasResults &r) { return r.mneut4(); })
      .def("mchg1", [](const MicromegasResults &r) { return r.mchg1(); })
      .def("mchg2", [](const MicromegasResults &r) { return r.mchg2(); })
      .def("mhsm", [](const MicromegasResults &r) { return r.mhsm(); })
      .def("mh", [](const MicromegasResults &r) { return r.mh(); })
      .def("mhc", [](const MicromegasResults &r) { return r.mhc(); });

  py::class_<SugraParameters>(m, "SugraParameters")
      .def(py::init<>(), "Construct an empty Parameters object with parameters "
                         "defined at the GUT "
                         "scale.")
      .def(py::init<double, double, double, double, double>(),
           "Construct a Parameters object with parameters "
           "defined at the GUT scale.",
           py::arg("m0"), py::arg("mhf"), py::arg("a0"), py::arg("tb"),
           py::arg("sgn"))
      .def_property(
          "m0", [](const SugraParameters &p) { return p.m0(); },
          [](SugraParameters &p) { return p.m0(); })
      .def_property(
          "mhf", [](const SugraParameters &p) { return p.mhf(); },
          [](SugraParameters &p) { return p.mhf(); })
      .def_property(
          "a0", [](const SugraParameters &p) { return p.a0(); },
          [](SugraParameters &p) { return p.a0(); })
      .def_property(
          "tb", [](const SugraParameters &p) { return p.tb(); },
          [](SugraParameters &p) { return p.tb(); })
      .def_property(
          "sgn", [](const SugraParameters &p) { return p.sgn(); },
          [](SugraParameters &p) { return p.sgn(); });

  py::class_<EwsbParameters>(m, "EwsbParameters")
      .def(
          py::init<>(),
          "Construct an empty Parameters object with parameters defined at the "
          "electroweak scale")
      .def(py::init<double, double, double, double, double, double, double,
                    double, double, double, double, double, double, double,
                    double, double, double, double, double, double, double,
                    double, double, double, double, double>(),
           "Construct a Parameters object with parameters defined at the "
           "electroweak scale.",
           py::arg("mu"), py::arg("mg1"), py::arg("mg2"), py::arg("mg3"),
           py::arg("ml1"), py::arg("ml2"), py::arg("ml3"), py::arg("mr1"),
           py::arg("mr2"), py::arg("mr3"), py::arg("mq1"), py::arg("mq2"),
           py::arg("mq3"), py::arg("mu1"), py::arg("mu2"), py::arg("mu3"),
           py::arg("md1"), py::arg("md2"), py::arg("md3"), py::arg("mh3"),
           py::arg("tb"), py::arg("at"), py::arg("ab"), py::arg("al"),
           py::arg("am"), py::arg("mtp"))
      .def_property(
          "mu", [](const EwsbParameters &p) { return p.mu(); },
          [](EwsbParameters &p) { return p.mu(); })
      .def_property(
          "mg1", [](const EwsbParameters &p) { return p.mg1(); },
          [](EwsbParameters &p) { return p.mg1(); })
      .def_property(
          "mg2", [](const EwsbParameters &p) { return p.mg2(); },
          [](EwsbParameters &p) { return p.mg2(); })
      .def_property(
          "mg3", [](const EwsbParameters &p) { return p.mg3(); },
          [](EwsbParameters &p) { return p.mg3(); })
      .def_property(
          "ml1", [](const EwsbParameters &p) { return p.ml1(); },
          [](EwsbParameters &p) { return p.ml1(); })
      .def_property(
          "ml2", [](const EwsbParameters &p) { return p.ml2(); },
          [](EwsbParameters &p) { return p.ml2(); })
      .def_property(
          "ml3", [](const EwsbParameters &p) { return p.ml3(); },
          [](EwsbParameters &p) { return p.ml3(); })
      .def_property(
          "mr1", [](const EwsbParameters &p) { return p.mr1(); },
          [](EwsbParameters &p) { return p.mr1(); })
      .def_property(
          "mr2", [](const EwsbParameters &p) { return p.mr2(); },
          [](EwsbParameters &p) { return p.mr2(); })
      .def_property(
          "mr3", [](const EwsbParameters &p) { return p.mr3(); },
          [](EwsbParameters &p) { return p.mr3(); })
      .def_property(
          "mq1", [](const EwsbParameters &p) { return p.mq1(); },
          [](EwsbParameters &p) { return p.mq1(); })
      .def_property(
          "mq2", [](const EwsbParameters &p) { return p.mq2(); },
          [](EwsbParameters &p) { return p.mq2(); })
      .def_property(
          "mq3", [](const EwsbParameters &p) { return p.mq3(); },
          [](EwsbParameters &p) { return p.mq3(); })
      .def_property(
          "mu1", [](const EwsbParameters &p) { return p.mu1(); },
          [](EwsbParameters &p) { return p.mu1(); })
      .def_property(
          "mu2", [](const EwsbParameters &p) { return p.mu2(); },
          [](EwsbParameters &p) { return p.mu2(); })
      .def_property(
          "mu3", [](const EwsbParameters &p) { return p.mu3(); },
          [](EwsbParameters &p) { return p.mu3(); })
      .def_property(
          "md1", [](const EwsbParameters &p) { return p.md1(); },
          [](EwsbParameters &p) { return p.md1(); })
      .def_property(
          "md2", [](const EwsbParameters &p) { return p.md2(); },
          [](EwsbParameters &p) { return p.md2(); })
      .def_property(
          "md3", [](const EwsbParameters &p) { return p.md3(); },
          [](EwsbParameters &p) { return p.md3(); })
      .def_property(
          "mh3", [](const EwsbParameters &p) { return p.mh3(); },
          [](EwsbParameters &p) { return p.mh3(); })
      .def_property(
          "tb", [](const EwsbParameters &p) { return p.tb(); },
          [](EwsbParameters &p) { return p.tb(); })
      .def_property(
          "at", [](const EwsbParameters &p) { return p.at(); },
          [](EwsbParameters &p) { return p.at(); })
      .def_property(
          "ab", [](const EwsbParameters &p) { return p.ab(); },
          [](EwsbParameters &p) { return p.ab(); })
      .def_property(
          "al", [](const EwsbParameters &p) { return p.al(); },
          [](EwsbParameters &p) { return p.al(); })
      .def_property(
          "am", [](const EwsbParameters &p) { return p.am(); },
          [](EwsbParameters &p) { return p.am(); })
      .def_property(
          "mtp", [](const EwsbParameters &p) { return p.mtp(); },
          [](EwsbParameters &p) { return p.mtp(); });
}

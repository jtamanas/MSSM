#ifndef PYMICROMEGAS_HPP
#define PYMICROMEGAS_HPP

#include <bitset>
#include <map>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <vector>

constexpr size_t IDX_MSNE = 0;  // Electron-snuetrino
constexpr size_t IDX_MSNM = 1;  // Muon-Snuetrino
constexpr size_t IDX_MSEL = 2;  // Left-handed Selectron
constexpr size_t IDX_MSER = 3;  // Right-handed Selectron
constexpr size_t IDX_MSML = 4;  // Left-handed Smuon
constexpr size_t IDX_MSMR = 5;  // Right-handed Smuon
constexpr size_t IDX_MSDL = 6;  // Left-handed down squark
constexpr size_t IDX_MSDR = 7;  // Right-handed up squark
constexpr size_t IDX_MSUL = 8;  // Left-handed up squark
constexpr size_t IDX_MSUR = 9;  // Right-handed up squark
constexpr size_t IDX_MSSL = 10; // Left-handed strange squark
constexpr size_t IDX_MSSR = 11; // Right-handed strange squark
constexpr size_t IDX_MSCL = 12; // Left-handed charm squark
constexpr size_t IDX_MSCR = 13; // Right-handed charm squark
constexpr size_t IDX_MSNL = 14; // Left-handed tau snuetrino
constexpr size_t IDX_MSL1 = 15; // Stau 1
constexpr size_t IDX_MSL2 = 16; // Stau 2
constexpr size_t IDX_MSB1 = 17; // Bottom squark 1
constexpr size_t IDX_MSB2 = 18; // Bottom squark 2
constexpr size_t IDX_MST1 = 19; // Top squark 1
constexpr size_t IDX_MST2 = 20; // Top squark 2
constexpr size_t IDX_MG = 21;   // Gluino mass
constexpr size_t IDX_MNE1 = 22; // Neutralino 1
constexpr size_t IDX_MNE2 = 23; // Neutralino 2
constexpr size_t IDX_MNE3 = 24; // Neutralino 3
constexpr size_t IDX_MNE4 = 25; // Neutralino 4
constexpr size_t IDX_MC1 = 26;  // Chargino 1
constexpr size_t IDX_MC2 = 27;  // Chargino 2
constexpr size_t IDX_MHSM = 28; // Higgs mass
constexpr size_t IDX_MH = 29;   // Other higgs mass
constexpr size_t IDX_MHC = 30;  // Charged Higgs mass

// ============================================================================
// ---- Settings --------------------------------------------------------------
// ============================================================================

class MicromegasSettings {
  static constexpr size_t NUM_BITS = 8;

  bool p_fast = true;
  double p_beps = 1e-4;
  double p_cut = 0.01;
  bool p_relic_density = true;
  bool p_masses = true;
  bool p_gmuon = true;
  bool p_bsg = true;
  bool p_bsmumu = true;
  bool p_btaunu = true;
  bool p_deltarho = true;
  bool p_sort_odd = true;

  static constexpr size_t BIT_RD = 0;
  static constexpr size_t BIT_MASSES = 1;
  static constexpr size_t BIT_BSG = 2;
  static constexpr size_t BIT_BSMUMU = 3;
  static constexpr size_t BIT_BTAUNU = 4;
  static constexpr size_t BIT_DELTARHO = 5;
  static constexpr size_t BIT_GMUON = 6;
  static constexpr size_t BIT_SORTODD = 7;

public:
  MicromegasSettings(bool relic_density = true, bool masses = true,
                     bool gmuon = true, bool bsg = true, bool bsmumu = true,
                     bool btaunu = true, bool delta_rho = true,
                     bool sort_odd = true, bool fast = true, double beps = 1e-4,
                     double cut = 1e-2) {
    p_relic_density = relic_density;
    p_masses = masses;
    p_gmuon = gmuon;
    p_bsg = bsg;
    p_bsmumu = bsmumu;
    p_btaunu = btaunu;
    p_deltarho = delta_rho;
    p_sort_odd = sort_odd;
    p_fast = fast;
    p_beps = beps;
    p_cut = cut;
  }

  auto fast() const -> const bool & { return p_fast; }
  auto beps() const -> const double & { return p_beps; }
  auto cut() const -> const double & { return p_cut; }
  auto compute_rd() const -> const bool & { return p_relic_density; }
  auto compute_masses() const -> const bool & { return p_masses; }
  auto compute_bsg() const -> const bool & { return p_bsg; }
  auto compute_bsmumu() const -> const bool & { return p_bsmumu; }
  auto compute_btaunu() const -> const bool & { return p_btaunu; }
  auto compute_deltarho() const -> const bool & { return p_deltarho; }
  auto compute_gmuon() const -> const bool & { return p_gmuon; }
  auto sort_odd_particles() const -> const bool & { return p_sort_odd; }

  auto fast() -> bool & { return p_fast; }
  auto beps() -> double & { return p_beps; }
  auto cut() -> double & { return p_cut; }
  auto compute_rd() -> bool & { return p_relic_density; }
  auto compute_masses() -> bool & { return p_masses; }
  auto compute_bsg() -> bool & { return p_bsg; }
  auto compute_bsmumu() -> bool & { return p_bsmumu; }
  auto compute_btaunu() -> bool & { return p_btaunu; }
  auto compute_deltarho() -> bool & { return p_deltarho; }
  auto compute_gmuon() -> bool & { return p_gmuon; }
  auto sort_odd_particles() -> bool & { return p_sort_odd; }
};

// ============================================================================
// ---- SUGRA Parameters ------------------------------------------------------
// ============================================================================

class SugraParameters {
  using value_type = double;
  value_type p_m0;
  value_type p_mhf;
  value_type p_a0;
  value_type p_tb;
  value_type p_sgn;
  // value_type p_mtp;
  // value_type p_mbmb;
  // value_type p_alfsmz;

  static constexpr size_t NUM_SUGRA_PARAMS = 8;
  static const std::array<std::string, NUM_SUGRA_PARAMS> SUGRA_PARAMS;

public:
  SugraParameters() {}

  SugraParameters(value_type m0, value_type mhf, value_type a0, value_type tb,
                  value_type sgn)
      : p_m0(m0), p_mhf(mhf), p_a0(a0), p_tb(tb), p_sgn(sgn) {}

  void initialize() const;

  auto m0() const -> const value_type & { return p_m0; }
  auto mhf() const -> const value_type & { return p_mhf; }
  auto a0() const -> const value_type & { return p_a0; }
  auto tb() const -> const value_type & { return p_tb; }
  auto sgn() const -> const value_type & { return p_sgn; }

  auto m0() -> value_type & { return p_m0; }
  auto mhf() -> value_type & { return p_mhf; }
  auto a0() -> value_type & { return p_a0; }
  auto tb() -> value_type & { return p_tb; }
  auto sgn() -> value_type & { return p_sgn; }
};

// ============================================================================
// ---- EWSB Parameters -------------------------------------------------------
// ============================================================================

class EwsbParameters {
  using value_type = double;
  value_type p_mu;
  value_type p_mg1;
  value_type p_mg2;
  value_type p_mg3;
  value_type p_ml1;
  value_type p_ml2;
  value_type p_ml3;
  value_type p_mr1;
  value_type p_mr2;
  value_type p_mr3;
  value_type p_mq1;
  value_type p_mq2;
  value_type p_mq3;
  value_type p_mu1;
  value_type p_mu2;
  value_type p_mu3;
  value_type p_md1;
  value_type p_md2;
  value_type p_md3;
  value_type p_mh3;
  value_type p_tb;
  value_type p_at;
  value_type p_ab;
  value_type p_al;
  value_type p_am;
  value_type p_mtp;

  static constexpr size_t NUM_EWSB_PARAMS = 26;
  static const std::array<std::string, NUM_EWSB_PARAMS> EWSB_PARAMS;

public:
  EwsbParameters() {}
  EwsbParameters(value_type mu, value_type mg1, value_type mg2, value_type mg3,
                 value_type ml1, value_type ml2, value_type ml3, value_type mr1,
                 value_type mr2, value_type mr3, value_type mq1, value_type mq2,
                 value_type mq3, value_type mu1, value_type mu2, value_type mu3,
                 value_type md1, value_type md2, value_type md3, value_type mh3,
                 value_type tb, value_type at, value_type ab, value_type al,
                 value_type am, value_type mtp)
      : p_mu(mu), p_mg1(mg1), p_mg2(mg2), p_mg3(mg3), p_ml1(ml1), p_ml2(ml2),
        p_ml3(ml3), p_mr1(mr1), p_mr2(mr2), p_mr3(mr3), p_mq1(mq1), p_mq2(mq2),
        p_mq3(mq3), p_mu1(mu1), p_mu2(mu2), p_mu3(mu3), p_md1(md1), p_md2(md2),
        p_md3(md3), p_mh3(mh3), p_tb(tb), p_at(at), p_ab(ab), p_al(al),
        p_am(am), p_mtp(mtp) {}

  auto mu() const -> const value_type & { return p_mu; }
  auto mg1() const -> const value_type & { return p_mg1; }
  auto mg2() const -> const value_type & { return p_mg2; }
  auto mg3() const -> const value_type & { return p_mg3; }
  auto ml1() const -> const value_type & { return p_ml1; }
  auto ml2() const -> const value_type & { return p_ml2; }
  auto ml3() const -> const value_type & { return p_ml3; }
  auto mr1() const -> const value_type & { return p_mr1; }
  auto mr2() const -> const value_type & { return p_mr2; }
  auto mr3() const -> const value_type & { return p_mr3; }
  auto mq1() const -> const value_type & { return p_mq1; }
  auto mq2() const -> const value_type & { return p_mq2; }
  auto mq3() const -> const value_type & { return p_mq3; }
  auto mu1() const -> const value_type & { return p_mu1; }
  auto mu2() const -> const value_type & { return p_mu2; }
  auto mu3() const -> const value_type & { return p_mu3; }
  auto md1() const -> const value_type & { return p_md1; }
  auto md2() const -> const value_type & { return p_md2; }
  auto md3() const -> const value_type & { return p_md3; }
  auto mh3() const -> const value_type & { return p_mh3; }
  auto tb() const -> const value_type & { return p_tb; }
  auto at() const -> const value_type & { return p_at; }
  auto ab() const -> const value_type & { return p_ab; }
  auto al() const -> const value_type & { return p_al; }
  auto am() const -> const value_type & { return p_am; }
  auto mtp() const -> const value_type & { return p_mtp; }

  auto mu() -> value_type & { return p_mu; }
  auto mg1() -> value_type & { return p_mg1; }
  auto mg2() -> value_type & { return p_mg2; }
  auto mg3() -> value_type & { return p_mg3; }
  auto ml1() -> value_type & { return p_ml1; }
  auto ml2() -> value_type & { return p_ml2; }
  auto ml3() -> value_type & { return p_ml3; }
  auto mr1() -> value_type & { return p_mr1; }
  auto mr2() -> value_type & { return p_mr2; }
  auto mr3() -> value_type & { return p_mr3; }
  auto mq1() -> value_type & { return p_mq1; }
  auto mq2() -> value_type & { return p_mq2; }
  auto mq3() -> value_type & { return p_mq3; }
  auto mu1() -> value_type & { return p_mu1; }
  auto mu2() -> value_type & { return p_mu2; }
  auto mu3() -> value_type & { return p_mu3; }
  auto md1() -> value_type & { return p_md1; }
  auto md2() -> value_type & { return p_md2; }
  auto md3() -> value_type & { return p_md3; }
  auto mh3() -> value_type & { return p_mh3; }
  auto tb() -> value_type & { return p_tb; }
  auto at() -> value_type & { return p_at; }
  auto ab() -> value_type & { return p_ab; }
  auto al() -> value_type & { return p_al; }
  auto am() -> value_type & { return p_am; }
  auto mtp() -> value_type & { return p_mtp; }

  void initialize() const;
};

void error_parser(const std::string &, int err);

// ============================================================================
// ---- Results ---------------------------------------------------------------
// ============================================================================

class MicromegasResults {
  using value_type = std::vector<double>;
  value_type p_omega{};
  value_type p_xf{};
  value_type p_bsgsm{};
  value_type p_bsgnlo{};
  value_type p_deltarho{};
  value_type p_bsmumu{};
  value_type p_btaunu{};
  value_type p_gmuon{};

  std::array<value_type, 31> p_masses;

  void sort_odd_particles(const MicromegasSettings &) const;
  void compute_relic_density(const MicromegasSettings &);
  void compute_masses(const MicromegasSettings &);
  void compute_bsg(const MicromegasSettings &);
  void compute_deltarho(const MicromegasSettings &);
  void compute_bsmumu(const MicromegasSettings &);
  void compute_btaunu(const MicromegasSettings &);
  void compute_gmuon(const MicromegasSettings &);

  void set_nans(size_t);

public:
  MicromegasResults(size_t n);

  void execute(const MicromegasSettings &);
  void execute(const MicromegasSettings &, const SugraParameters &);
  void execute(const MicromegasSettings &, const EwsbParameters &);

  auto omega() const -> const value_type & { return p_omega; }
  auto xf() const -> const value_type & { return p_xf; }
  auto b_sg_sm() const -> const value_type & { return p_bsgsm; }
  auto b_sg_nlo() const -> const value_type & { return p_bsgnlo; }
  auto delta_rho() const -> const value_type & { return p_deltarho; }
  auto b_smumu() const -> const value_type & { return p_bsmumu; }
  auto b_taunu() const -> const value_type & { return p_btaunu; }
  auto g_muon() const -> const value_type & { return p_gmuon; }
  auto msne() const -> const value_type & { return p_masses[IDX_MSNE]; }
  auto msnm() const -> const value_type & { return p_masses[IDX_MSNM]; }
  auto msel() const -> const value_type & { return p_masses[IDX_MSEL]; }
  auto mser() const -> const value_type & { return p_masses[IDX_MSER]; }
  auto msml() const -> const value_type & { return p_masses[IDX_MSML]; }
  auto msmr() const -> const value_type & { return p_masses[IDX_MSMR]; }
  auto msdl() const -> const value_type & { return p_masses[IDX_MSDL]; }
  auto msdr() const -> const value_type & { return p_masses[IDX_MSDR]; }
  auto msul() const -> const value_type & { return p_masses[IDX_MSUL]; }
  auto msur() const -> const value_type & { return p_masses[IDX_MSUR]; }
  auto mssl() const -> const value_type & { return p_masses[IDX_MSSL]; }
  auto mssr() const -> const value_type & { return p_masses[IDX_MSSR]; }
  auto mscl() const -> const value_type & { return p_masses[IDX_MSCL]; }
  auto mscr() const -> const value_type & { return p_masses[IDX_MSCR]; }
  auto msnl() const -> const value_type & { return p_masses[IDX_MSNL]; }
  auto msl1() const -> const value_type & { return p_masses[IDX_MSL1]; }
  auto msl2() const -> const value_type & { return p_masses[IDX_MSL2]; }
  auto msb1() const -> const value_type & { return p_masses[IDX_MSB1]; }
  auto msb2() const -> const value_type & { return p_masses[IDX_MSB2]; }
  auto mst1() const -> const value_type & { return p_masses[IDX_MST1]; }
  auto mst2() const -> const value_type & { return p_masses[IDX_MST2]; }
  auto mg() const -> const value_type & { return p_masses[IDX_MG]; }
  auto mneut1() const -> const value_type & { return p_masses[IDX_MNE1]; }
  auto mneut2() const -> const value_type & { return p_masses[IDX_MNE2]; }
  auto mneut3() const -> const value_type & { return p_masses[IDX_MNE3]; }
  auto mneut4() const -> const value_type & { return p_masses[IDX_MNE4]; }
  auto mchg1() const -> const value_type & { return p_masses[IDX_MC1]; }
  auto mchg2() const -> const value_type & { return p_masses[IDX_MC2]; }
  auto mhsm() const -> const value_type & { return p_masses[IDX_MHSM]; }
  auto mh() const -> const value_type & { return p_masses[IDX_MH]; }
  auto mhc() const -> const value_type & { return p_masses[IDX_MHC]; }

  auto omega() -> value_type & { return p_omega; }
  auto xf() -> value_type & { return p_xf; }
  auto bsgsm() -> value_type & { return p_bsgsm; }
  auto bsgnlo() -> value_type & { return p_bsgnlo; }
  auto deltarho() -> value_type & { return p_deltarho; }
  auto bsmumu() -> value_type & { return p_bsmumu; }
  auto btaunu() -> value_type & { return p_btaunu; }
  auto gmuon() -> value_type & { return p_gmuon; }
  auto msne() -> value_type & { return p_masses[IDX_MSNE]; }
  auto msnm() -> value_type & { return p_masses[IDX_MSNM]; }
  auto msel() -> value_type & { return p_masses[IDX_MSEL]; }
  auto mser() -> value_type & { return p_masses[IDX_MSER]; }
  auto msml() -> value_type & { return p_masses[IDX_MSML]; }
  auto msmr() -> value_type & { return p_masses[IDX_MSMR]; }
  auto msdl() -> value_type & { return p_masses[IDX_MSDL]; }
  auto msdr() -> value_type & { return p_masses[IDX_MSDR]; }
  auto msul() -> value_type & { return p_masses[IDX_MSUL]; }
  auto msur() -> value_type & { return p_masses[IDX_MSUR]; }
  auto mssl() -> value_type & { return p_masses[IDX_MSSL]; }
  auto mssr() -> value_type & { return p_masses[IDX_MSSR]; }
  auto mscl() -> value_type & { return p_masses[IDX_MSCL]; }
  auto mscr() -> value_type & { return p_masses[IDX_MSCR]; }
  auto msnl() -> value_type & { return p_masses[IDX_MSNL]; }
  auto msl1() -> value_type & { return p_masses[IDX_MSL1]; }
  auto msl2() -> value_type & { return p_masses[IDX_MSL2]; }
  auto msb1() -> value_type & { return p_masses[IDX_MSB1]; }
  auto msb2() -> value_type & { return p_masses[IDX_MSB2]; }
  auto mst1() -> value_type & { return p_masses[IDX_MST1]; }
  auto mst2() -> value_type & { return p_masses[IDX_MST2]; }
  auto mg() -> value_type & { return p_masses[IDX_MG]; }
  auto mneut1() -> value_type & { return p_masses[IDX_MNE1]; }
  auto mneut2() -> value_type & { return p_masses[IDX_MNE2]; }
  auto mneut3() -> value_type & { return p_masses[IDX_MNE3]; }
  auto mneut4() -> value_type & { return p_masses[IDX_MNE4]; }
  auto mchg1() -> value_type & { return p_masses[IDX_MC1]; }
  auto mchg2() -> value_type & { return p_masses[IDX_MC2]; }
  auto mhsm() -> value_type & { return p_masses[IDX_MHSM]; }
  auto mh() -> value_type & { return p_masses[IDX_MH]; }
  auto mhc() -> value_type & { return p_masses[IDX_MHC]; }
};

#endif // !PYMICROMEGAS_HPP

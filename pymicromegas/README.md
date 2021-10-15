# `pymicromegas`

## Installation

Copy the Makefile `Makefile.pymicromegas` and the `pymicromegas` directory into the MSSM directory of your `micromegas` installation
(located at `micromegas/MSSM`.) Then execute the following:

```sh
make --file=Makefile.pymicromegas
```

This will generate the following set of python modules:
- `pymicromegas`: module containing classes for storing results, specifying parameters and settings,
- `softsusy`: module containing a single function called `softsusy` which uses SoftSusy as the RGE/spectrum generation backend,
- `spheno`: same as `softsusy` but uses `SPheno` as the backend,
- `suspect`: again the same, but uses `SuSpect` as the backend.

These module can be used by copying the modules to a directory where you plan to use them or by appending you python path
to include the `micromegas/MSSM` directory:

```python
import sys
sys.append("path/to/micromegas/MSSM")
```

## Usage

### Basic

First, load the relevant packages:

```python
# If you left the modules in micromegas/MSSM:
import sys
sys.append("path/to/micromegas/MSSM")

from pymicromegas import EwsbParameters, MicromegasSettings, SugraParameters
from softsusy import softsusy
from spheno import spheno
from suspect import suspect
```

The `SugraParameters` class is used to specify parameters defined at the GUT scale:

```python
sugra = SugraParameters(m0=9158.4500, mhf=8913.2500, a0=-14770.2000, tb= 40.6663, sgn=1)
```

To run `micromegas`, we can then use:

```python
# Using SoftSusy
res_softsusy = softsusy(sugra)

# Using SPheno
res_spheno = spheno(sugra)

# Using SuSpect
res_suspect = suspect(sugra)
```

### Accessing Results

The result of `softsusy`, `suspect` or `spheno` is a `MiMicromegasResults` object. The results may be accessed
using `result.param()` where `param` is the result you wish to access. For example, to access the relic-density
and g-2 of the muon, use:

```python
sugra = SugraParameters(m0=9158.4500, mhf=8913.2500, a0=-14770.2000, tb= 40.6663, sgn=1)
results = softsusy(sugra)

# Get the relic density:
results.omega()
# Get the g-2 of the muon:
results.gmuon()
```

`results.param()` will *always* return a list (since it is possible to specify a batch of parameters at a time
and we want to keep things type-stable.) The full list of available results are:

```python
results.omega() # Dark matter relic density scaled by h^2: Omega h^2
results.xf() # freeze-out value of x = (dark-matter mass) / (temperature)
results.bsgsm() # Partial width of B -> s + gamma (SM contribution)
results.bsgnlo() # Partial width of B -> s + gamma at NLO
results.deltarho() # Delta-rho
results.bsmumu() # Partial width of B -> s + mu + mu
results.btaunu() # Partial width of B -> tau + nu
results.gmuon() # g-2 of the muon
results.msne() # Electron-snuetrino
results.msnm() # Muon-Snuetrino
results.msel() # Left-handed Selectron
results.mser() # Right-handed Selectron
results.msml() # Left-handed Smuon
results.msmr() # Right-handed Smuon
results.msdl() # Left-handed down squark
results.msdr() # Right-handed up squark
results.msul() # Left-handed up squark
results.msur() # Right-handed up squark
results.mssl() # Left-handed strange squark
results.mssr() # Right-handed strange squark
results.mscl() # Left-handed charm squark
results.mscr() # Right-handed charm squark
results.msnl() # Left-handed tau snuetrino
results.msl1() # Stau 1
results.msl2() # Stau 2
results.msb1() # Bottom squark 1
results.msb2() # Bottom squark 2
results.mst1() # Top squark 1
results.mst2() # Top squark 2
results.mg() # Gluino mass
results.mneut1() # Neutralino 1
results.mneut2() # Neutralino 2
results.mneut3() # Neutralino 3
results.mneut4() # Neutralino 4
results.mchg1() # Chargino 1
results.mchg2() # Chargino 2
results.mhsm() # Standard-Model Higgs mass
results.mh() # Other Higgs mass
results.mhc() # Charged Higgs mass
```

### Running a batch of parameters

You can run a batch of parameters at once by passing a list of `SugraParameters` or `EwsbParameters`:

```python
sugras = [
  SugraParameters( 3.74707e+03, 2.03617e+02, -3.57260e+03, 4.77960e+01, -1.00000e+00 )
  SugraParameters( 9.15845e+03, 8.91325e+03, -1.47702e+04, 4.06663e+01, 1.00000e+00  )
  SugraParameters( 9.65301e+01, 1.24578e+03, -5.69820e+02, 3.37576e+01, -1.00000e+00 )
  SugraParameters( 5.65860e+03, 9.32066e+03, 2.43138e+04,  2.28000e+01, -1.00000e+00 )
  SugraParameters( 3.36232e+03, 3.36401e+03, 3.21739e+03,  2.35756e+01, -1.00000e+00 )
]

results = softsusy(sugras)
```

### Using settings

You can alter how `micromegas` is run by passing a `MicromegasSettings` object:
```python
# These are the default settings:
settings = MicromegasSettings(
    relic_density=True, # Compute relic density?
    masses=True,        # Compute mass spectrum?
    gmuon=True,         # Compute muon g-2?
    bsg=True,           # Compute b -> s + gamma ?
    bsmumu=True,        # Compute b -> s + mu + mu?
    btaunu=True,        # Compute b -> tau + nu?
    delta_rho=True,     # Compute delta-rho?
    sort_odd=True,      # Compute delta-rho?
    fast=True,          # Compute relic-density using a approximation? (see note on relic-density)
    beps=0.0001,        # Set value for Beps (see note on relic-density)
)

# Run with settings
res_softsusy = softsusy(sugra, settings)
```

### Note on Relic Densities

The `fast` and `beps` parameters control how the relic-density is computed. See this excerpt form the `micrOmegas` manual:

> **darkOmega(&Xf, fast, Beps)**
> > calculates the dark matter relic density Ωh2. This routine solves the differential evolution
> > equation using the Runge-Kutta method. Xf = M cdm/Tf characterizes the freeze-out
> > temperature. The value of Xf is given for information and is also used as an input for the
> > routine that gives the relative contribution of each channel to Ωh2, see printChannels
> > below. The f ast = 1 flag forces the fast calculation (for more details see Ref. [2]). This is
> > the recommended option and gives an accuracy around 1%. The parameter Beps defines
> > the criteria for including a given coannihilation channel in the computation of the ther-
> > mally averaged cross-section, [2]. The recommended value is Beps = 10−4 −10−6 whereas
> > if Beps = 1 only annihilation of the lightest odd particle is computed.

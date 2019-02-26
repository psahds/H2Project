H2Project

This code performs enemble MCMC sampling on 3 scaling relations for local galaxies:

1. Star formation rate (SFR) vs. H2 mass
2. Star formation rate vs. HI mass
3. Stellar mass vs. star formation rate

This is part of a larger study that attempts to determine the cosmic abundance of molecular hydrogen (H2) in the local universe, by building an H2 mass function for a group of local galaxies. We aim to accomplish this by inferring the H2 mass function from the known stellar mass function for local galaxies, using scaling relations that take us from stellar mass to H2 mass (from stellar mass we can obtain SFR using relation 3, and from SFR we can obtain H2 mass using relation 1). The same methos is applied to HI mass since the HI mass function is well determined locally by the ALFALFA survey (https://arxiv.org/pdf/1802.00053.pdf). 

This study makes use of the xCOLD GASS survey (http://www.star.ucl.ac.uk/xCOLDGASS), the xGASS survey (http://xgass.icrar.org/data.html) and the GAMA survey (http://www.gama-survey.org/).

This code makes use of the Python package emcee. To download, follow instructions on: http://dfm.io/emcee/current/#

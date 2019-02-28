# H2 Project

This code performs enemble MCMC sampling on 3 scaling relations for local galaxies:

1. Star formation rate (SFR) vs. H2 mass
2. Star formation rate vs. HI mass
3. Stellar mass vs. star formation rate

This is part of a larger study that attempts to determine the cosmic abundance of molecular hydrogen (H2) in the local universe, by building an H2 mass function for a group of local galaxies. We aim to accomplish this by inferring the H2 mass function from the known stellar mass function for local galaxies, using scaling relations that take us from stellar mass to H2 mass (from stellar mass we can obtain SFR using relation 3, and from SFR we can obtain H2 mass using relation 1). The same method is applied to HI since the HI mass function is well determined locally by the ALFALFA survey (https://arxiv.org/pdf/1802.00053.pdf), so it serves as a way to validate this method. 

This study makes use of the xCOLD GASS survey (http://www.star.ucl.ac.uk/xCOLDGASS), the xGASS survey (http://xgass.icrar.org/data.html) and the GAMA survey (http://www.gama-survey.org/). The data from these surveys that was used in this study can be found online at the given links, and is also in the Survey_Data folder.

This code makes use of the Python packages emcee (http://dfm.io/emcee/current/#) and corner (https://corner.readthedocs.io/en/latest/index.html).

The functions and main program can all be found in scalingrelations.py, output figures can be found in repository as yielded from code, in .pdf format.

# upchannel

This project trains deep learning models to read in mono audio files and output stereo files. The models themselves are extensible to other input and output channel configurations, including mono -> 5.1, stereo -> 5.1, as well as others.

This repo contains several models, using both spectral domain and waveform domain based systems. In particular, there are pure u-net based models (heavily based on the models implemented by Milesial and jaxony), as well as GANs that utilize the aforementioned u-nets as generators.

These models work on various datasets, but I have found that a dataset consisting of contemporary A Cappella music best suits the models, as modern techniques in A Cappella recording and production place a much heavier emphasis on width and surround sound than other genres of music (and their associated engineering standards) that can use real instruments for breadth of timbre. To that end, I have trained these models on the following dataset:

Lithium (Faux Paz- University of Maryland)

Panorama (Reverb- Florida State University)

In Full Color (YellowJackets- University of Rochester)

Come Together (M-Pact)

PTX Vol. 2 (Pentatonix)


Each sampled at 16khz, and fed into the models in ~4 second chunks. This process is completed by db_splitter.py

Each model has a train_(net type)_(domain).py and a network_(net type)_(domain).py file. In all cases, train_(...).py is dependent on network_(...).py.

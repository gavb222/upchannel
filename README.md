# upchannel

This project trains deep learning models to read in mono audio files and output stereo files. The models themselves are extensible to other input and output channel configurations, including mono -> 5.1, stereo -> 5.1, as well as others.

This repo contains several models, using both spectral domain and waveform domain based systems. In particular, there are pure u-net based models (heavily based on the models implemented by Milesial and jaxony), as well as GANs that utilize the aforementioned u-nets as generators.

I have trained this model on several datasets ranging from pop music, to contemporary a cappella music, to string quartets. I have found that since string quartets feature not only different timbres of instruments, but also a common standard for mixing and recording setup that facilitates spatial differentiation, they are the best domain of music to train the model on. To that end, I have trained the model on several albums by the St. Lawrence String Quartet, specifically their rendition of Haydn string quartet op. 20, coming in at a full runtime of 2:33:00. I plan on extending this dataset to more audio soon. For validation data, I used the Borodin string quartet playing Alexander Borodin's string quartet in D major, coming in at 8:37.

Each sampled at 16khz, and fed into the models in ~4 second chunks. This process is completed by db_splitter.py

Each model has a train_(net type)_(domain).py and a network_(net type)_(domain).py file. In all cases, train_(...).py is dependent on network_(...).py.

I have determined that while the spectral domain models work, they are severely limited by their need to convert magnitude spectrograms back to the waveform domain, which is an intensely noise-producing process. Thus, I have narrowed my focus to the waveform domain based models, which is why there is a test_waveform.py but not a test_spectral.py in this repo.

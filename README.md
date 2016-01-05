# Evo_NN

Evolutionary algorithm in conjunction with PyBrain neural network as fitness evaluator

First runs maino to evolve noise through NN operating on pcm

Two methods of encoding songs. Raw pcm data and fourier decomposition of signal.

Trains net on dataset of encoded songs and random noise. Neural network is used to evaluate signals, evolutionary algorithm selects and
reproduces signals with best fitness rating, as given by trained neural network. 

Output is given as population of evolved arrays of pcm data or fourier coefs.

Songs are encoded directly from desktop and are given from lines 620 to end.

Multi-state (dyanmical?) systems analysis

### General idea
I've been assumming early/late (expecting/not expecting change) corresponds to two distinct states. 
But this doesn't really explain a few things, including why psychometrics/chronometrics barely change, and why 
animal's behaviour is so unpredictable in general.
So: maybe its more like there's some set of distinct states (e.g. lick-permissive, sensorimotor transformation 
permissive, lick-supressing) that the brain oscillates between - and these are independent to block. Rather, what 
changes might be occupancy in each of these states - i.e. only sometimes are animals in sensorimotor-transformation 
permissive states, but with a higher probability if expectation is high...


### framework
big picture: each state = a dynamical system (linear?); thinking some mish-mash of k-means-like iterative updating of 
DSs to best captue potential states. 
obvious caveat - probably WAY too slow practically, and not sure if there's some circularity here. 

...and how to choose number of states? as unsupervised as poss would be good... need a principled way to do this.

maybe can actually just start with literally k means on tf-aligned psths? but included pulses near licks?

#### scratch

so - get tf outliers -> kmeans - on pre-pulse state? pulse state? 
just visualize occpancy map in pc space? i.e. trial trajectories -> heatmap?

maybe just start with plotting single trial trajectories. smooth ramp predicting lickiness or fluctuation between 
states? how to extract dimension corresponding to lick propensity...? just window for now?

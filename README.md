# CTBench

CTBench is a software, developed in Python, intending to uniformly collect a set of methods to be used as Benchmarking in our experiments, particularly regarding Sparse-view Computed Tomography. In the following, we denote by $K \in \mathbb{R}^n$ the linear operator representing the Discrete Radon Transform, $y^\delta$ is the acquired noisy sinogram, defined as:

$$
y^\delta = K x^{GT} + e,
$$

where $|| e ||_2 \leq \delta$ and $e$ is sampled by a Gaussian distribution.
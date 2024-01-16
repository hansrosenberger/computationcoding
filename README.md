# Repository for Computation Coding Algorithms
Contains the Python implementation of linear computation coding (LCC) algorithms presented in [1-4]. These algorithms target the optimization of constant matrix vector multiplication problems. 
LCC algorithms are particularly well suited for an implementation on reconfigurable hardware, e.g. field programmable gate arrays (FPGAs)[5].

### Structure of the repository

#### Algorithms
Contains the following CMVM decomposition algorithms:

- dmp.py: Discrete Matching Pursuit algorithm (DMP) [1]
- rs.py: Reduced-State (RS) algorithm [3]
- lz.py: Fully Sequential (FS) algorithm/Lempel-Ziv inspired algorithm [2]  
- graph.py: Mixed algorithm (MA) [4]
- parallel_graph.py: Mixed algorithm (MA) constrained to a fully parallel structure [4]
- wiring_red.py: Methods to recursively remove unused computations/unused DAG vertices from LCC matrix products
- lz_vis.py: functions for directed acyclic graph (DAG) visualization and operations on DAGs
- mcm.py: MCM algorithms presented in [6] adapted to CMVM by applying the MCM algorithms columnwise (requires a compiled version of the C++ code in [6] available from https://spiral.ece.cmu.edu/mcm/gen.html)
- wrappers.py: wrapper functions for of algorithms for parallelization

### Dependencies
Following dependencies required (tested with the given versions):
- Python 3.9.13
- numpy 1.22.3
- scipy 1.8.0
- matplotlib 3.8.1
- networkx 2.8.5 
- graphviz 5.0
- ray 2.8.1

### References

[1] R. Müller, B. Gäde, A. Bereyhi, 'Linear computation coding: A framework for joint quantization and computing', Algorithms, vol. 15, no. 7, p. 253, 2022

[2] R. Müller, 'Linear computation coding inspired by the Lempel-Ziv algorithm', Information Theory Workshop (ITW), IEEE, 2022

[3] H. Rosenberger, J. Fröhlich, A. Bereyhi, R. Müller, 'Linear computation coding: Exponential search and reduced-state algorithms', Data Compression Conference (DCC), IEEE, 2023

[4] H. Rosenberger, A. Bereyhi, R. Müller, 'Graph-based algorithms for Linear Computation Coding', International Zürich Seminar (IZS) on Information and Communication, 2024

[5] A. Lehnert, P. Holzinger, S. Pfennig, R. Müller, M. Reichenbach, 'Most resource efficient matrix vector multiplication on FPGAs', IEEE Access, vol. 11, pp. 3881-3898, 2023

[6] Y. Voronenko, M. Püschel, 'Multiplierless multiple constant multiplication', ACM Transactions on Algorithms, vol. 3, no. 2, 2007

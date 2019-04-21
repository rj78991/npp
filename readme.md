## Reference Implementation of Parallel, One-Pass Computation of Statistical Moments 

A reference impplementation of SAND2018-6212 (see docs) for Robust, One-Pass Parallel Computation of Covariances and Arbitrary Order Statistical Moments.

The computation is implemented using Intel C++ Threading Building Blocks for multi-core parallelism. The original algorithm is adapted for map-reduce on symmetric multi-cores. Use on a distributed cluster is faciliated by exposing the single-node building block via a python binding for use with large scale map-reduce frameworks such as Hadoop.

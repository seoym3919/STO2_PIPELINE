# STO2_PIPELINE
Data pipeline for Stratospheric Terahertz Observatory 2 (STO2).

STO2 is an astronomical observatory on a balloon platform to observe C+, N+, and O in the interstellar space. The data contains the spectra of N+ and C+ recored in one-dimentaional 1024 channels.

This repo includes the python scripts of the data pipeline for STO2. This is the pipeline for automatically reducing entire raw STO2 data to Level 1. For more controlled/detailed reduction, use package in STO2 repo.

The data pipeline includes a decision tree using DBSCAN, a defringing algorithm using a deflational ICA, and a baseline correction algorithm using a parallel ICA, which are machine learning algorithms and highly non-convetional in astronomy. The pipeline is finely tuned for STO2 data, so please use it carefully if you are using it for data other than STO2.   

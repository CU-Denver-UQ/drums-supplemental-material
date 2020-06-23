[![DOI](https://zenodo.org/badge/274504322.svg)](https://zenodo.org/badge/latestdoi/274504322)

# Supplemental material for the article "What do we hear from a drum? A data-consistent approach to quantifying irreducible uncertainty on model inputs by extracting information from correlated model output data" by Troy Butler and Harri Hakula

The supplemental material in this repository allows the interested reader to re-create all of the figures and table data presented in the article.
Here, we provide a brief description of the data formats and software dependencies required to run the scripts.

All eigenmode data are stored in `.mat` formats containing descriptive variable names for the various arrays of data in each file.
The Python module, `DrumAnalysis`, and scripts provided as supplemental material to this article were developed using Python 3.7.3 and the following libraries

- `scipy` (version 1.2.1)
- `numpy` (version 1.16.2)
- `scikit-learn` (version 0.20.3)
- `matplotlib` (version 3.0.3)

The instructions below assume an installation of Python with the appropriate dependencies as described above and that all supplemental material are located within the same path (directory).
Then, from a terminal or command prompt, navigate to the appropriate path containing all the files.
The two main scripts are `Example-1.py` and `Example-2.py` that will generate all of the relevant figures and table data appearing in the Numerical Results section (section 5) in the article.
Depending on machine specifications and user-edits to either script, they may both take several minutes to execute. 

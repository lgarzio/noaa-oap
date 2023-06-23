# NOAA - Ocean Acidification Program
Tools for manipulating and plotting pH glider data. 

Author: Lori Garzio (lgarzio@marine.rutgers.edu)

## Installation Instructions
Add the channel conda-forge to your .condarc. You can find out more about conda-forge from their website: https://conda-forge.org/

`conda config --add channels conda-forge`

Clone the noaa-oap repository.

`git clone https://github.com/lgarzio/noaa-oap.git`

Change your current working directory to the location that you downloaded noaa-oap. 

`cd /Users/lgarzio/Documents/repo/noaa-oap/`

Create conda environment from the included environment.yml file:

`conda env create -f environment.yml`

Once the environment is done building, activate the environment:

`conda activate noaa-oap`

Install the toolbox to the conda environment from the root directory of the noaa-oap toolbox:

`pip install .`

The toolbox should now be installed to your conda environment.

## Glider Data
Full-resolution delayed-mode glider datasets containing raw pH voltages can be found on [RUCOOL's Glider ERDDAP Server](http://slocum-data.marine.rutgers.edu/erddap/index.html).

Full-resolution quality-controlled delayed-mode glider datasets containing calculated pH and other variables can be found on the [Glider DAC's ERDDAP Server](https://gliders.ioos.us/erddap/index.html).

## Citations
[CODAP-NA](https://essd.copernicus.org/articles/13/2777/2021/): Jiang, L.-Q., Feely, R. A., Wanninkhof, R., Greeley, D., Barbero, L., Alin, S., Carter, B. R., Pierrot, D., Featherstone, C., Hooper, J., Melrose, C., Monacci, N., Sharp, J. D., Shellito, S., Xu, Y.-Y., Kozyr, A., Byrne, R. H., Cai, W.-J., Cross, J., Johnson, G. C., Hales, B., Langdon, C., Mathis, J., Salisbury, J., and Townsend, D. W.: Coastal Ocean Data Analysis Product in North America (CODAP-NA) – an internally consistent data product for discrete inorganic carbon, oxygen, and nutrients on the North American ocean margins, Earth Syst. Sci. Data, 13, 2777–2799, https://doi.org/10.5194/essd-13-2777-2021, 2021.

Humphreys, M. P., Gregor, L., Pierrot, D., van Heuven, S. M. A. C., Lewis, E. R., and Wallace, D. W. R. (2020). [PyCO2SYS](https://pypi.org/project/PyCO2SYS/): marine carbonate system calculations in Python. Zenodo. doi:10.5281/zenodo.3744275.

Lewis, E. and Wallace, D. W. R. (1998) Program Developed for CO2 System Calculations, ORNL/CDIAC-105, Carbon Dioxide Inf. Anal. Cent., Oak Ridge Natl. Lab., Oak Ridge, Tenn., 38 pp., [https://salish-sea.pnnl.gov/media/ORNL-CDIAC-105.pdf](https://salish-sea.pnnl.gov/media/ORNL-CDIAC-105.pdf)
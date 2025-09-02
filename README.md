# Solar Wind Speed Mapping from Metis Data
During my B.Sc. Thesis I got interested in the calculation of the Electron density maps from Solar Corona 
remote observation through the as old as effective van de Hulst inversion technique. 

Following this technique, the Electron density is computed from manipulating the polarized Brightness (pB)
images. All the algorithm have been tested on ESA-NASA's Solar Orbiter/Metis coronagraph images.

In this repository is contained the natural extension of this work, with the calculation of the Solar wind 
speed outflow velocities maps from Electron density maps. The software lets you download a pB and and
UltraViolet (UV) image from the Solar Orbiter ARchive (SOAR) through the 'notebooks/SOAR_downloader.ipynb' 
notebook, selecting a given time window of interest.

Then, in the 'notebooks/Metis_from_pB_and_UV_to_Wind_Speed.ipynb' is possible to invert the pB image to 
retrieve the Electron density map, and after combining it with the UV image following a 2-step iterative
procedure, it is possible to obtain the outflow velocity according to which the Intensity synthesized by 
the coronal emission model is closer (within 1% tolerance) to the observed one.

The calculation process was validated through the comparision of data outputs with a previous exisiting
version (already validated) written in IDL. The comparison between the two outputs can be found in 
'notebooks/validation/DDT_test_&_validation.ipynb'.

Python_Doppler_Dimming_Technique/
├── data/ # Dataset e file di input
│ ├── input/  # Input data, e.g., downloaded through 'SOAR_downloader.ipynb'
│ │ ├── UV_IMAGES/ # contains UltraViolet Ly-alpha L2 images
| │ └── VL_IMAGES/ # contains polarized Brightness L2 images
│ └── output/ # Output data: computed Solar Wind speed maps
│
├── notebooks/ # Jupyter Notebooks
│ ├── Metis_from_pB_and_UV_to_Wind_Speed.ipynb # Compute new maps
│ └── SOAR_downloader.ipynb # Download L2 images from Solar Orbiter ARchive (SOAR)
│
├── src/ 
│ ├── aux_lib_lyman_alpha.py # contains the "new" functions necessary for the implementation of the Doppler Dimming Technique
│ └── metis_aux_lib.py # contains the functions for Metis data manipulation and Electron density calculation; based on the version from A. Burtovoi
│
├── requirements.txt # Required python libraries, open to check installation instruction
└── README.md # This file

## 📧 Contact

For any questions, or if you plan to use this software, feel free to contact me:  
**dario.vetrano@studenti.polito.it**

I will be happy to discuss the code, highlight its strengths, and point out its current limitations :).

---

⚠️ *This is a research-oriented project and still under development. Use with care and validate results accordingly.*

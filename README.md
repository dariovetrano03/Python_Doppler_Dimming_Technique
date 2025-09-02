# Solar Wind Speed Mapping through Doppler Dimming Technique
During my B.Sc. thesis, I became interested in calculating electron density maps of the solar corona from remote observations using the **van de Hulst inversion technique**, a method as old as effective.

Using this technique, the electron density is derived from the manipulation of polarized brightness (pB) images.

This repository contains a natural extension of that work: the calculation of **solar wind outflow velocity maps**. The software allows you to download pB and ultraviolet (UV) images from the Solar Orbiter Archive (SOAR) via the `notebooks/SOAR_downloader.ipynb` notebook, selecting a specific time window of interest.

Next, in `notebooks/Metis_from_pB_and_UV_to_Wind_Speed.ipynb`, the pB images can be inverted to retrieve the electron density map. By combining this map with the UV images through a two-step iterative procedure, the outflow velocity can be obtained, such that the intensity synthesized by the coronal emission model matches the observed intensity within a 1% tolerance.

The calculation process was validated by comparing the outputs with a previous, already validated version written in IDL. The comparison can be found in `notebooks/validation/DDT_test_&_validation.ipynb`.


```text
Python_Doppler_Dimming_Technique/
├── data/ # Dataset e file di input
│   ├── input/  # Input data, e.g., downloaded through `SOAR_downloader.ipynb'
│   │   ├── UV_IMAGES/ # contains UltraViolet Ly-alpha L2 images
│   │   └── VL_IMAGES/ # contains polarized Brightness L2 images
│   └── output/ # Output data: computed Solar Wind speed maps
│
├── notebooks/ # Jupyter Notebooks
│   ├── Metis_from_pB_and_UV_to_Wind_Speed.ipynb # Compute new maps
│   └── SOAR_downloader.ipynb # Download L2 images from Solar Orbiter ARchive (SOAR)
│
├── src/ 
│   ├── aux_lib_lyman_alpha.py # contains the "new" functions necessary for the implementation of the Doppler Dimming Technique
│   └── metis_aux_lib.py # contains the functions for Metis data manipulation and Electron density calculation; based on the version from A. Burtovoi
│
├── requirements.txt # Required python libraries, open to check installation instruction
└── README.md # This file
```



## 📧 Contact

For any questions, or if you plan to use this software, feel free to contact me:  
**dario.vetrano@studenti.polito.it**

I will be happy to discuss the code, highlight its strengths, and point out its current limitations :).

---

⚠️ *This is a research-oriented project and still under development. Use with care and validate results accordingly.*

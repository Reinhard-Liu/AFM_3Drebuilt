e-cienciaDatos: "readme" form 

-------------------
GENERAL INFORMATION
-------------------
1. Title of dataset: QUAM-AFM Lite


2. Contact

   Principal Investigator Contact Information
        Name: Rubén Pérez
        Institution: Universidad Autónoma de Madrid (UAM)
        Email: ruben.perez@uam.es
        ORCID: 0000-0001-5896-541X

3. Description of the project

Artificial Intelligence (AI) is a lively, thriving field with many applications and an outstanding future. Among the different AI techniques, Deep Learning is nowadays the standard tool for image and automatic speech recognition, drug discovery, medical image analysis and bioinformatics. Furthermore, AI is also changing Basic Sciences: evolution of the galaxies, design of new materials or determination of quantum wave functions are only few of thousands other examples. In this project, we propose to exploit recent Deep Learning developments on an emerging and very promising field: Automatic identification of molecules based on High-Resolution Atomic Force Microscopy (HR-AFM).  

Molecules are key materials for Nanotechnology. Based on their properties, determined by their composition and structure, new powerful devices can be designed including organic photovoltaics, light-emitting diodes, transistors, sensors, actuators and supercapacitors. The key issue in all these cases is the enhanced performance associated with the tunability of the electronic properties that can be achieved through the control of the atomic structure and the chemical composition of the molecules. Here, radical new concepts as on-surface chemistry, have been added to the classical solution chemistry in order to synthesize molecules tailored to function.

Atomic force microscopy (AFM) [Binnig1986] in combination with dynamic operation [Garcia2002,Giessibl2003] has become one of the key tools for imaging and manipulation of materials and biological systems at the nanoscale. Dynamic AFM senses the effect on the dynamics of an oscillating cantilever of the interaction between a sharp tip mounted at the end of the cantilever and the sample under observation. With this information, an image of the sample, reflecting its structure and composition is formed. It took almost a decade for one of the dynamic modes, frequency-modulation (FM) AFM, commonly known as non-contact (NCAFM), to fulfill the goal of achieving atomic resolution. HR-AFM represents the latest NCAFM breakthrough: the use of metal tips functionalized with a CO molecule at the tip apex, has provided access to the internal structure of molecules with totally unprecedented resolution [Gross2009,Gross2018]. 

HR-AFM resolves intermolecular features and bond orders in aromatic compounds [Gross2018], and can be used to induce on-surface chemical reactions [Pavlicek2017]. Its capability to address individual molecules has paved the way for the identification of natural compounds (where conventional methods failed) and of intermediates and final products generated in on-surface reactions, and to resolve a hundred different types of molecules in complex mixtures such as asphaltenes [Gross2018]. However, the unambiguous identification of the structure and composition of individual molecules from AFM images, without any prior information, remains a formidable challenge.

We plan to tackle this problem combining our ability to accurately simulate HR-AFM images with the gathering of large data set of simulated images to exploit Deep Learning techniques. The use of theoretical models to simulate HR-AFM images has become essential to understand the contrast observed in the experiments and to link it with the structure and chemical composition of the molecule. Moreover, the large data sets needed for the training cannot be provided by the experiments, and reliable theoretical simulations have to take the lead.

[Binnig1986] Binnig, G.; Quate, C.F.; Gerber, C. Atomic force microscope. Phys. Rev. Lett. 1986, 56, 930–933. http://doi.org/10.1103/PhysRevLett.56.930
 
[Garcia2002] García, R.; Pérez, R. Dynamic atomic force microscopy methods. Surf. Sci. Rep. 2002, 47, 197–301. http://dx.doi.org/10.1016/S0167-5729(02)00077-8

[Giessibl2003] Giessibl, F.J. Advances in atomic force microscopy. Rev. Mod. Phys. 2003, 75, 949. http://dx.doi.org/10.1103/RevModPhys.75.949

[Gross2009] Gross, L.; Mohn, F.; Moll, N.; Liljeroth, P.; Meyer, G. The Chemical Structure of a Molecule Resolved by Atomic Force Microscopy. Science 2009, 325, 1110–1114. http://dx.doi.org/10.1126/science.1176210

[Gross2018] Gross, L.; Schuler, B.; Pavlicek, N.; Fatayer, S.; Majzik, Z; Moll, N.; Peña, D.; Meyer, G. Atomic Force Microscopy for Molecular Structure Elucidation. Angew. Chem.Int. Ed. 2018, 57, 3888-3908.  https://doi.org/10.1002/anie.201703509

[Pavlicek2017] Pavlicek, N.; Gross, L. Generation, manipulation and characterization of molecules by atomic force microscopy. Nat. Rev. Chem.
2017, 1, 0005. https://doi.org/10.1038/s41570-016-0005


4. Description of the dataset

QUAM–AFM Lite is the scaled-down version of QUAM-AFM, the largest dataset of simulated Atomic Force Microscopy (AFM) images. This reduced version was generated from a selection of 1,755 molecules that span the most relevant bonding structures and chemical species in organic chemistry. Similar to the extended version, QUAM-AFM Lite contains, for each molecule,  24 3D image stacks, each consisting of constant-height images simulated for 10 tip-sample distances (in the relevant imaging range and spanning a variation of 1 Å (0.1 nanometers)) with one of the 24 different combination of AFM operational parameters, resulting in a total of 421,200 images with a resolution of 256x256 pixels. 

The operational parameters include six different values for the cantilever oscillation amplitude (0.40, 0.60, 0.80, 1.00, 1.20, 1.40Å), 4 values of the elastic constant describing the tilting of the CO tip (0.40, 0.60, 0.80 and 1.00 N/m). The first parameter is freely chosen in the experiments in order to enhance different features of the image, while the last one reflects differences in the attachment of the CO molecule to the metal tip that are routinely observed and has been characterized in the experiments.

The data provided for each molecule includes, besides a set of AFM images, the ball–and–stick depiction, the IUPAC name, the chemical formula, the atomic coordinates, and the map of atom heights. In order to simplify the use of the collection as a source of information, we have developed a Graphical User Interface (GUI) that allows the search for structures by CID number, IUPAC name or chemical formula.

This dataset arises as a product of the research carried out in collaboration between Quasar Science Resources S.L. (https://quasarsr.com) and the Scanning Probe Microscopy Theory & Nanomechanics Research Group (SPMTH) (http://www.uam.es/spmth) at the Universidad Autónoma de Madrid (UAM), funded by the Comunidad de Madrid under the Industrial Doctorate Programme 2017 (project reference IND2017/IND-7793).

The main goal of this dataset is to provide a simplified version of QUAM-AFM that allows to analyse the distribution of information and/or the graphical interface without the need for a full download. The extended version, QUAM-AFM, supports the development of deep learning methods for molecular identification through AFM imaging. Once this project has concluded, this dataset is made freely accessible in order to facilitate and to promote research in a range of fields including Atomic Force Microscopy, on-surface synthesis and deep learning applications.

5. Notes

QUAM-AFM Lite is a small part of the whole QUAM-AFM dataset intended for testing purposes. The dataset has been developed in the SPMTH Group (www.uam.es/spmth) at the Universidad Autónoma de Madrid. The simulations for QUAM-AFM have been carried out on the Finisterrae II supercomputer of the Centro de Supercomputación de Galicia (CESGA) with a total of 2.5 million computing hours provided by the Red Española de Supercomputación (RES).

6. Deposit date

November 2021

7. Date
--N/A

8. Language: English


--------------------------
AUTHOR INFORMATION
--------------------------
1. Authors

        Name: Jaime
        Last name: Carracedo-Cosme
        Institution: Quasar Science Resources S.L.
        Email: jcarracedo@quasarsr.com

        Name: Carlos
        Last name: Romero-Muñíz
        Institution: Universidad de Sevilla
        Email: crm1988@hotmail.com

        Name: Pablo
        Last name: Pou
        Institution: Universidad Autónoma de Madrid
        Email: pablo.pou@uam.es

        Name: Rubén
        Last name: Pérez
        Institution: Universidad Autónoma de Madrid
        Email: ruben.perez@uam.es

--------------------------
METHODOLOGY
--------------------------
1. Methodology

In order to obtain a sufficiently large set of molecular structures, we have carried out a massive, systematic download of the atomic coordinates of ``3D conformers'' available on the PubChem website (https://pubchem.ncbi.nlm.nih.gov), identifying each of these structures by the CID number associated with it on this website. Using this label, we can simplify the search of molecules, especially when two structures have the same atoms. QUAM-AFM Lite provides a Python dictionary in which it identifies the chemical formula with the CID number of each compound. 

We filtered the molecular structures on the basis of several criteria that make them of special interest for AFM research. First, we restricted basically to organic molecules, discarding all other compounds that may not have purely molecular forms, like organic salts or inorganic compounds. Therefore, we have selected only the molecules containing the four basic elements of organic chemistry (carbon, hydrogen, nitrogen and oxygen) plus some other less common elements which are still frequent on organic compounds like sulfur, phosphorus and the halogen atoms (fluorine, chlorine, bromine and iodine). Then, we imposed two restrictions on the size of the molecules. On the one hand, we discarded very small molecules, namely, those containing less than eight atoms. These molecules are extremely mobile and display a huge variety of adsorption configurations due to their small size. Therefore, they are not good candidates to be identified solely by means of AFM. In addition, we discarded very large molecules, having a structure that does not fit into a square-based cell with a side length of 24Å. We imposed this restriction for two reasons: larger unit cells will dramatically increase the computational cost, and we want to use the same unit cell for all the molecules in order to avoid either the repetition of the calculation for the CO tip in each different unit cell  or a cumbersome process of padding to adapt the tip calculation for a small unit cell to larger cells. The largest molecule in QUAM-AFM Lite has a total of eighty-five atoms. 

Although these restrictions may seem to be strong, our criteria leads to a large, representative set of molecules that includes aliphatic, cyclic and aromatic compounds. In particular, we can find a large number of hydrocarbons (alkanes, alkenes, alkynes, etc.) together with all the traditional organic families (alcohols, thiols, ethers, aldehydes and ketones, carboxylic acids, amines, amides, imines, esters, nitriles, nitro and azo compounds, halocarbons and acyl halides, etc.). Besides being specially appropriate for AFM characterization, this set is particularly relevant for on-surface chemistry, a powerful alternative to traditional synthesis methods based on solution chemistry that constitutes a  very active research field.

The use of deep learning techniques for AFM image-based molecular identification has to face two main challenges that are intrinsic to the technique: how to achieve chemical identification at the single atom level, and how to deal with markedly non-planar structures. For this reason, we have restricted our selection for the database to quasi-planar molecules, that is, molecules which display only height variations up to 1.83Å along the z-axis. In spite of the restrictions, we are still left with a huge dataset of more than 1,755 molecules, significantly larger than those used in previous deep learning works in the field and more importantly, that spans relevant structural and compositional moieties in organic chemistry, and, particularly, in the field of on-surface synthesis. 

The HR-AFM simulations of QUAM-AFM Lite have been generated with an approximate implementation of the FDBM method [Ellner2019], FDBM@PPM, that is available in the latest release [Liebig2020 of the PPM suite of codes [Hapala2014]. The FDBM method splits the total tip-sample interaction in four contributions: short range (SR), electrostatic (ES), van de Waals (vdW), and an harmonic contribution in the tilting angle that accounts for the CO flexibility. The first two contributions can be efficiently determined from the total charge densities of the tip and the sample, (rho_tip) and (rho_sample), and the electrostatic potential of the sample (phi_sample), obtained from two independent ab initio simulations. The FDBM@PPM implementation makes further approximations for the computation of the SR, ES and vdW contributions, that are described in references [Liebig2020, Carracedo2021a, Carracedo2021b], in order to speed up the calculations. For a given tip's initial position R_tip, the probe's relaxed coordinates are obtained by minimizing the potential V(R_tip,theta,psi) with respect to the tilt polar theta and azimuth psi angles.

The total charge densities of the tip and the sample, and the electrostatic potential of the sample have been calculated using the state-of-the-art ab-initio code VASP [VASp]. VASP transforms the many-body electronic structure problem of a periodic system into an effective one-body problem using the Density Functional Theory (DFT), solving the Kohn-Sham equations using pseudo-potentials to describe the electron-ion interaction and a plane-wave basis.  

[Ellner2019] Ellner, M.; Pavliˇcek, N.; Pou, P.; Schuler, B.; Moll, N.; Meyer, G.; Gross, L.; Pérez, R. The Electric Field of CO Tips and Its Relevance
for Atomic Force Microscopy. Nano Lett. 2016, 16, 1974–1980. https://doi.org/10.1021/acs.nanolett.5b05251

[Liebig2020] Liebig, A.; Hapala, P.; Weymouth, A.J.; Giessibl, F.J. Quantifying the evolution of atomic interaction of a complex surface with a
functionalized atomic force microscopy tip. Sci. Rep. 2020, 10, 14104.  https://doi.org/10.1038/s41598-020-71077-9

[Hapala2014] Hapala, P.; Kichin, G.;Wagner, C.; Tautz, F.S.; Temirov, R.; Jelínek, P. Mechanism of high-resolution STM/AFM imaging with
functionalized tips. Phys. Rev. B 2014, 90, 085421. https://doi.org/10.1103/PhysRevB.90.085421

[Carracedo2021a] Carracedo-Cosme, J.; Romero-Muñiz, C.; Pérez, R. A Deep Learning Approach for Molecular Classification Based on AFM Images. Nanomaterials 2021, 11, 1658. https:// doi.org/10.3390/nano11071658

[Carracedo2021b] Carracedo-Cosme, J.; Romero-Muñiz, C.; Pou, P. ; Pérez, R. QUAM–AFM: a Free Database for Molecular Identification by Atomic Force Microscopy (submitted) 2021


2. Software

 - Vienna Ab initio Simulation Package (VASP, https://www.vasp.at/) for calculations of the charge density and electrostatic potential for each of the molecules.

 - FDBM@PPM (an approximate implementation of the FDBM model in the Probe Particle Model (PPM) suite of code) for the simulation of AFM images using the data calculated with VASP. FDBM@PPM is described in: https://doi.org/10.1038/s41598-020-71077-9

 
--------------------------
KEYWORDS
--------------------------
1. Keywords

Condensed Matter Physics, Organic Chemistry, Machine Learning

--------------------------
SPONSORSHIP INFORMATION AND GRANT IDs
--------------------------
1. Grant Information.

Funded by the Comunidad de Madrid under the Industrial Doctorate Programme 2017 (project reference IND2017/IND-7793).  We also acknowledge support from the Spanish MINECO (project MAT2017-83273-R) and from the Spanish Ministry of Science and Innovation (MICINN) (projects PID2020-115864RB-I00 and CEX2018-000805-M). Computer time provided by the Red Española de Supercomputación (RES) at the Finisterrae II Supercomputer (Centro de Supercomputación de Galicia, CESGA) is gratefully acknowledged.

--------------------------
RELATED PUBLICATIONS
--------------------------
1. Related publication

Carracedo-Cosme, J.; Romero-Muñiz, C.; Pou, P. ; Pérez, R. QUAM–AFM: a Free Database for Molecular Identification by Atomic Force Microscopy (submitted) 2021

2. Related dataset

QUAM-AFM. DOI: https://doir.org/10.21950/UTGMZ7

--------------------------
GEOGRAPHIC INFORMATION
--------------------------
1. Spatial coverage

--N/A

--------------------------
TEMPORAL INFORMATION
--------------------------
1. Time period coverage

--N/A

--------------------------
FILES
--------------------------
1. Files 
SUBMIT_QUAM_AFM.tar.gz : QUAM-AFM Lite contains simulated AFM images for 1755 molecules. These images are located in the QUAM folder. The GUI folder contains a Graphical User Interface (GUI) that allows visualization and search of the data in QUAM, using the Python pickle module dictionaries in the DICTIONARIES folder

QUAM-AFM Lite has a layout that is particularly suitable for use in both training deep neural networks and querying simulated AFM images. The QUAM-AFM Lite dataset is divided into three sub-folders: QUAM, GUI and DICTIONARIES.

The AFM images are distributed in 24 folders (K-1,...,K-24), within the "QUAM" directory, according to the different combinations of simulation parameters (six values for the cantilever oscillation amplitude and four values for the CO tip elasticity stiffness). In each of these 24 folders there are 1,755 subfolders identifying the 10 AFM simulations at different tip-sample distances of each molecule based on the CID number provided by PubChem. The QUAM directory contains three additional directories in addition to those for the simulations. In these, a ball-and-stick representation of the molecules (JMOL_IMAGES), a height map (HM) and the molecular configuration used for the simulation in xyz format (XYZ_FILES) are provided.

The GUI directory contains the executable QUAM.py, which displays a graphical interface which provides easy access to both the AFM images and the graphical descriptors of each molecule, and allows a quick search by the CID number, the composition or the IUPAC name. 

The DICTIONARIES directory contains dictionaries generated with the pickle module of Python 3.7 that is used by the graphical interface to search for images or provide additional information. These dictionaries (all_CIDS.pkl, CID_FORMULA.pkl, CID_IUPAC.pkl, Folds_Params.pkl, FORMULA_CIDS.pkl and IUPAC_CID.pkl) contain the CID numbers identifying the molecules, the chemical formula, the IUPAC name and the simulation parameters used.

--------------------------
LICENSES AND PRIVACITY
--------------------------
1. Licenses (+ info http://www.consorciomadrono.es/investigam/licencias/)

The data can be reused under the terms of the Creative Commons CC-BY-NC-SA license. https://creativecommons.org/licenses/by-nc-sa/4.0/


--------------------------
OTHERS
--------------------------
1. Data dictionary

The data can be read both with the GUI and with the Python dictionaries provided where the CID number of each molecule is related to the corresponding IUPAC name.

2. DOI: 10.21950/BFAU11


# PREWS5 - Open Data Curation and Image Analysis in Digital Rocks Portal

Tutorial files and notebooks for the Dec. 8, 2024 workshop on open data curation and image analysis in Digital Rocks Portal, presented at AGU24.

### Presenters:
- Masa Prodanovic
- Maria Esteva
- Richard A Ketcham
- Bahareh Nojabaei
- Bernard Chang
- Cinar Turhan




## Workshop Description
Computed tomography (CT), micro-computed tomography (μCT), and focused ion beam scanning electron microscopy (FIB-SEM), are now applied routinely to acquire three-dimensional images that reveal the structure of geologic materials.  3D imaging has allowed many porous media processes to be observed and simulated in detail, providing key insights into the mechanisms that directly impact their behaviors at larger scales.
This  workshop will instruct researchers in how to curate 3D image data in the re-designed open data repository [Digital Rocks Portal](https://www.digitalrocksportal.org/) as well as perform image analysis for characterizing digitized porous materials and for creating simulation inputs. Curation will involve identifying and documenting the image material properties according to descriptive standards as well as documenting the image quality.  Analyses The analysis portion will include hands-on practice using Jupyter Notebook workflows that can be run on a personal laptop or used as part of the suite of open source applications available in the Digital Rocks Portal. Researchers will get practical experience in analysis/visualization of 2D and 3D images, quantifying properties such as porosity, tortuosity, Minkowski functionals, and estimating data heterogeneity as well as their associated uncertainty. This requires a combination of advanced image analysis algorithms, scientific visualization and computation. This workshop is sponsored in part by NSF GEO OSE grant #2324786. We gratefully acknowledge the high performing computing systems that will be provided by the Texas Advanced Computing Center available through the Digital Rocks Portal.

*Software Use and Requirements:* The workshop will be a combination of presentations, literature overview and hands-on exercises for at least one third of the class time. Attendees will be expected to bring their  laptops for hands-on exercises. Visualization and image analysis software used in class include Python 3 programming environment (Anaconda distribution) for advanced exercises and we will clearly communicate installation instructions prior to the workshop. All the open source code used in the workshop will be shared via GitHub.. We do not assume working/programming knowledge for any of these software packages beyond basic familiarity with programming basics in Python (or similar language); all further guidance will be provided during class.
These scripts will be used as preprocessing quality checks on data to be used for federated learning. These contain some ideas of metrics that clients are permitted to share about the training data to ensure data quality and representativeness.

This workshop is organized into Jupyter notebooks and Python scripts including topics on computing the Minkowski Functionals (MFs), heterogeneity characterization, and competent subset selection.

## Workshop Agenda
The workshop will be a combination of presentations, literature overview and hands-on exercises for at least one third of the class time. Attendees will be encouraged to bring their own datasets and if they want to publish data in Digital Rocks Portal, we will provide immediate feedback on curating datasets and best practices. Alternatively, we will have practice data examples selected from Digital Rocks Portal. As specified in Description, attendees will also leave with open source code for all of the concepts learned in the workshop, thus ensuring they achieve the learning outcomes. The proposed example agenda is as follows:

**9-10:15am:** (Part 1) Digital Rocks Portal: overview of new functionality and metadata model followed by a hands-on exercise in uploading and documenting a dataset.

**10:15-10:25am:** Break

**10:25- 11:15am:** (Part 2) Fundamentals of grayscale image filtering and image segmentation including hands-on exercises.

**11:15-11:25am:** Break

**11:25-12:30am:** (Part 3) Geometric characterization of segmented images and assessing their suitability for further simulation. Hands-on exercises (Python) follow each of the topics:

- Quantifying Minkowski functionals
- Measuring heterogeneity
- Selecting a competent and/or representative subset for simulation and visualization
- Morphological drainage for assessing ample image resolution



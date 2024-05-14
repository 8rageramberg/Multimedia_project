# Multimedia project: Group 4
## DOMAIN SPECIFIC VIDEO RETRIEVAL FOR STRENGTH, HYPERTROPHY AND CONDITIONING
An innovative and technological way of detecting your exercise, without the need for a personal trainer. This is done via utilizing reverse image searching with the assistance of classification models.

&nbsp;


### Prerequisites:
**Dependencies:**
You need the following dependencies to run our application:

*OpenCV*
> -  *OpenCV:* a realtime computer vision library:
	- Pip install: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `pip install opencv-python`

&nbsp;

  *MediaPipe*
>-  *MediaPipe:* Google's open source framework for building ML pipeline
		- Pip install: `pip install --upgrade mediapipe`

&nbsp;

  *NumPy*
>-  *NumPy:* A package for scientific computing in python. Used mostly for the np.ndarray datastructure.
		- Pip install: `pip install --upgrade numpy`

&nbsp;

*Scipy*
> -  *Scipy:* Python library for mathematical computations and operations ( e.g. euclidean distance)
	-Pip install: `pip install --upgrade scipy``

&nbsp;

  *Flask*
> -  *Flask:* Python library for simple web app development and
	- Pip install: `pip install --upgrade flask`

&nbsp;

  *Keras*
> -  *Keras:* A open-source deep learning library utilized for the CNN
	- Pip install: ` pip install --upgrade keras`

&nbsp;

  *Sklearn*
> -  *Sklearn:* Sci-kit learns own library for machine learning.
	- Pip install: ` pip install --upgrade sklearn`

&nbsp;

  *Tensorflow*
> -  *Tensorflow:* Tensorflows own library for machine learning.
	- Pip install: ` pip install --upgrade tensorflow`

&nbsp;

  *Pandas*
> -  *Pandas:* Pandas is a tool for data manipulation, such as making dataframes and saving pickle files.
	- Pip install: ` pip install --upgrade pandas`
&nbsp;

&nbsp;

**Feature Database as a .pkl file:**
To actually be able to run the application you need the pickle files associated with each feature extractor. These files contain the features of each and every image in our database, stored as a pickle file for fast and convenient retrieval when running the program.

These can be downloaded from [here](https://unisydneyedu-my.sharepoint.com/:f:/g/personal/thus0518_uni_sydney_edu_au/EkcsZbO2zZhAi57Mvj_Z-9YBKax0SaSjgek4x6M04-BWtQ?e=L2eJcV).
You need to put these three files into the folder feature_db.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *NB: Link expires on the 13th of may 2025!*

&nbsp;

&nbsp;

**Installation:**

To install our application you can do the following steps:

1. Open up your preferred editor.
2. In the CLI, navigate to your preferred directory and run the following command to clone the repository: `git clone git@github.com:8rageramberg/Multimedia_project.git` (Via SSH).
3. After running this command you should have successfully cloned the repository.

NB: If you're having trouble cloning the repositroy refer to [githubs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) own documentation on cloning repositories.

&nbsp;

&nbsp;

### Usage:
To use our application, open up the cloned repository in your editor of choice. Afterwards, navigate yourself to the `main.py` file, which should be located at the root level of the repository:

![README_IMG_1](https://github.com/8rageramberg/Multimedia_project/blob/main/readme_imgs/readme_img1.png)

Afterwards, run the repository and open a new browser and type in `http://127.0.0.1:8000`

![README_IMG_2](https://github.com/8rageramberg/Multimedia_project/blob/main/readme_imgs/readme_img2.png)

(or alternatively go to the terminal in your editor and do the following:)

- MAC: `CMD` + `left click` the link in the terminal
- Windows / Linux: `CTRL` + `left click` the link in the terminal

![README_IMG_3](https://github.com/8rageramberg/Multimedia_project/blob/main/readme_imgs/readme_img3.png)

Now the application should have started and you should be shown the following webpage:

![README_IMG_4](https://github.com/8rageramberg/Multimedia_project/blob/main/readme_imgs/readme_img4.png)

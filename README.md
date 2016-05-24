# Buil-Research-Envirorment-with-Tensorflow-OpenCV-Python
(Version 0.1, Last Update 24/05/2016)

Tutorial on how to build your own research envirorment for Deep Learning with OpenCV, Python, Tensorfow on Linux Machine.


The Project  follow the below **index**:

1. **Introduction to the Problem;**
2. **Possible Solutions;**
3. **Virtual Environment**
      1. **Virtual Environment & Python;**
      2. **OpenCV;**
      3. **Tensorflow;**
      4. **CUDA;**
      5. **Other Libraries;**
4. **Conclusions;**
5. **References;**
6. **Copyright**



##1. Introduction to the Problem

Nowadays, the Deep Learning Area use lots of  different libraries, dependicies, script languages, some of the most known are:
- Caffè (http://caffe.berkeleyvision.org/);
- Tensorflow (https://www.tensorflow.org/);
- Keras (http://keras.io/);
- OpenCV (http://opencv.org/).

So Develop & Research on Deep Learning, now means struggle with lots of error, compilation, installation, and some times most of them comes from updates, new library dependecies, let's see which are the possible solutions.

##2. Possible Solutions

Mainly we have three solutions:
- Install on own machine;
- Install in a server (Use the environment remotly);
- Install on Virtual Environment.
This last one is the one, covered in this Tutorial.

##3. Virtual Environment

**Why Virtual Environment?**

Why is easy, Virtual Envirorments lets us to have different dependencies, libraries, updates, of the same or different software, without struggling with error due to mislink or new apis.

The tutorial code consider you are the owner and have all grants to run **sudo** , if you  wanna apply the tutorial to a server you have to ask your administrator to install OpenCV & CUDA on the machine and give you the path to export into the bashrc file, for VirtualEnv, Tensorflow, Python nothing change.

Let's see a four step set up installation for our Research Envirorment with Tensorflow, OpenCV, Python, CUDA and all the libraries that we could need:

### i.Virtual Environment & Python

Open up a terminal and update the apt-get package manager followed by upgrading any pre-installed packages:
```
$ sudo apt-get update 
$ sudo apt-get upgrade

```
Now we need to install our developer tools:
```
$ sudo apt-get install build-essential cmake git pkg-config
```

The pkg-config is likely already installed, but be sure to include it just in case. 
We’ll be using git to pull down the OpenCV repositories from GitHub. 
The cmake package is used to configure our build.

OpenCV needs to be able to load various image file formats from disk, including JPEG, PNG, TIFF, etc. In order to load these image formats from disk, we’ll need our image I/O packages:
```
$ sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
```
At this point, we have the ability to load a given image off of disk. But how do we display the actual image to our screen? The answer is the GTK development library, which the highgui  module of OpenCV depends on to guild Graphical User Interfaces (GUIs):
```
$ sudo apt-get install libgtk2.0-dev
```
We can load images using OpenCV, but what about processing video streams and accessing individual frames? We’ve got that covered here:
```
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
```
Install libraries that are used to optimize various routines inside of OpenCV:
```
$ sudo apt-get install libatlas-base-dev gfortran
```
Install pip, a Python package manager:
```
$ wget https://bootstrap.pypa.io/get-pip.py $ sudo python get-pip.py
```

Install virtualenv and virtualenvwrapper.These two packages allow us to create separate Python environments for each project we are working on. While installing virtualenv  and virtualenvwrapper is not a requirement to get OpenCV 3.0 and Python 2.7+ up and running on your Ubuntu system, I highly recommend it and the rest of this tutorial will assume you have them installed!
```
$ sudo pip install virtualenv virtualenvwrapper 
$ sudo rm -rf ~/.cache/pip
```
Now that we have virtualenv  and virtualenvwrapper  installed, we need to update our ~/.bashrc  file:
```
$ export WORKON_HOME=$HOME/.virtualenvs source /usr/local/bin/virtualenvwrapper.sh
```
This quick update will ensure that both virtualenv  and virtualenvwrapper  are loaded each time you login.
To make the changes to our ~/.bashrc  file take effect, you can either (1) logout and log back in, (2) close your current terminal window and open a new one, or preferably, (3) reload the contents of your ~/.bashrc  file:
```
$ source ~/.bashrc
```
Lastly, we can create our cv  virtual environment where we’ll be doing our computer vision development and OpenCV 3.0 + Python 2.7+ installation:
```
$ mkvirtualenv cv
```
As I mentioned above, this tutorial covers how to install OpenCV 3.0 and Python 2.7+, so we’ll need to install our Python 2.7 development tools:
```
$ sudo apt-get install python2.7-dev
```
Since OpenCV represents images as multi-dimensional NumPy arrays, we better install NumPy into our cv  virtual environment:
```
$ pip install numpy
```
### ii.OpenCV
Our environment is now all setup, we can proceed to change to our home directory, pull down OpenCV from GitHub, and checkout the 3.0.0  version:
```
$ cd ~ $ git clone https://github.com/Itseez/opencv.git
$ cd opencv 
$ git checkout 3.0.0
```
>You can replace the 3.0.0  version with whatever the current release is (as of right now, it’s 3.1.0 ). 
>Be sure to check OpenCV.org for information on the latest release.

We also need the opencv_contrib repo as well. Without this repository, we won’t have access to standard keypoint detectors and local invariant descriptors (such as SIFT, SURF, etc.) that were available in the OpenCV 2.4.X version. We’ll also be missing out on some of the newer OpenCV 3.0 features like text detection in natural images:
```
$ cd ~ 
$ git clone https://github.com/Itseez/opencv_contrib.git 
$ cd opencv_contrib 
$ git checkout 3.0.0
```
Again, make sure that you checkout the same version for opencv_contrib that you did for opencv above, otherwise you could run into compilation errors.
Time to setup the build:

```
$ cd ~/opencv 
$ mkdir build 
$ cd build 
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \ -D CMAKE_INSTALL_PREFIX=/usr/local \ -D INSTALL_C_EXAMPLES=ON \ -D INSTALL_PYTHON_EXAMPLES=ON \ -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \ -D BUILD_EXAMPLES=ON ..
```

> Building instructions for OSX, more specified parameters:
>$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \-D PYTHON2_PACKAGES_PATH=~/.virtualenvs/cv/lib/python2.7/site-packages \-D PYTHON2_LIBRARY=/usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7/bin \-D PYTHON2_INCLUDE_DIR=/usr/local/Frameworks/Python.framework/Headers \-D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON \-D BUILD_EXAMPLES=ON \-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules ..

>The compiling options changes depending on the version of python and most depending on how we are installing all the libraries and the dependencies.


> In order to build OpenCV 3.1.0, you need to set -D INSTALL_C_EXAMPLES=OFF (rather than ON) in the cmake command. There is a bug in the OpenCV v3.1.0 CMake build script that can cause errors if you leave this switch on. Once you set this switch to off, CMake should run without a problem.

Now we can finally compile OpenCV:
```
$ make -j4
```
Where you can replace the 4 with the number of available cores on your processor to speedup the compilation.
				
**OpenCV Linking**: If we have installed OpenCV globally on your PC and you want to link it to the virtualenv, we need to export the two following paths:
- ** export PYTHONPATH="${PYTHONPATH}:/my/other/path"**
(Example of path: /opt/amd64/opencv-3.1.0/lib/python2.7/dist-packages)
- ** export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/my/other/path"**
(Example of path: /opt/amd64/opencv-3.1.0/lib)


Assuming that OpenCV compiled without error, you can now install it on your Ubuntu system:
```
$ sudo make install 
$ sudo ldconfig
```
If you’ve reached this step without an error, OpenCV should now be installed in ** /usr/local/lib/python2.7/site-packages**.
However, our cv virtual environment is located in our home directory — thus to use OpenCV within our cv environment, we first need to sym-link OpenCV into the site-packages directory of the cv virtual environment:
```
$ cd ~/.virtualenvs/cv/lib/python2.7/site-packages/ 
$ ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
```
Congratulations! You have successfully installed OpenCV 3.0 with Python 2.7+ bindings on your Ubuntu system!
To confirm your installation, simply ensure that you are in the cv virtual environment, followed by importing cv2:
```
$ workon cv 
$ python 
```
Your output should be:
```
>>> import cv2
>>> cv2.__version__ '3.0.0'
```

### iii.Tensorflow
  
### iv.CUDA
  
### v.Other Libraries
  
##4. Conclusions
##5. References
##6. Copyright

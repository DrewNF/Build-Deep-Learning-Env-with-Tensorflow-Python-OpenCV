# Build Deep Learning Env with Tensorflow Python OpenCV
(Version 0.1, Last Update 24/05/2016)

Tutorial on how to build your own research envirorment for Deep Learning with OpenCV, Python, Tensorfow on Linux Machine.
This Repository try to be a clear summary of the many guides you can find online (I will link in Referencies all the guides, I used to compose this Repository-Tutorial)

The Project follow the below **index**:

1. **[Introduction to the Problem](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#1-introduction-to-the-problem);**
2. **[Possible Solutions](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#2-possible-solutions);**
3. **[Virtual Environment](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#3-virtual-environment)**
      1. **[Virtual Environment & Python](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#ivirtual-environment--python);**
      		i.**[Preparing Linux Machine](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#preparing-linux-machine);**
		ii.**[Preparing Macintosh OSX Machine](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#preparing-macintosh-osx-machine);**
		iii.**[Set Up the Virtual Env](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#set-up-the-virtual-env);**
      2. **[OpenCV](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#iiopencv);**
      3. **[Tensorflow](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#iiitensorflow);**
      4. **[CUDA](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#ivcuda);**
      5. **[Other Libraries](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#vother-libraries);**
4. **[Conclusions](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#4-conclusions);**
5. **[References](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#5-references);**
6. **[Copyright](https://github.com/DrewNF/Build-Deep-Learning-Env-with-Tensorflow-Python-OpenCV/blob/master/README.md#6-copyright).**



##1. Introduction to the Problem

Nowadays, the Deep Learning Area use lots of  different libraries, dependicies, script languages, some of the most known are:
- [Caffè](http://caffe.berkeleyvision.org/);
- [Tensorflow](https://www.tensorflow.org/);
- [Keras](http://keras.io/);
- [OpenCV](http://opencv.org/).

So Develop & Research on Deep Learning, now means struggle with lots of error, compilation, installation, and some times most of them comes from updates, new library dependecies, let's see which are the possible solutions.

##2. Possible Solutions

Mainly we have three solutions:
- Install on own machine;
- Install in a server (Use the environment remotly);
- Install on Virtual Environment.
This last one is the one, covered in this Tutorial.

##3. Virtual Environment

**Why Virtual Environment?**

Why is easy: **Virtual Envirorments** lets us to have different dependencies, libraries, updates, of the same or different software, without struggling with error due to mislink or new apis.

The tutorial code consider you are the owner and have all grants to run **sudo** , if you  wanna apply the tutorial to a server you have to ask your administrator to install OpenCV & CUDA on the machine and give you the path to export into the bashrc file, for VirtualEnv, Tensorflow, Python nothing change.

Let's see a four step set up installation for our Research Envirorment with Tensorflow, OpenCV, Python, CUDA and all the libraries that we could need:

### i.Virtual Environment & Python

#### i.Preparing Linux Machine

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
$ wget https://bootstrap.pypa.io/get-pip.py 
$ sudo python get-pip.py
```
#### ii.Preparing Macintosh OSX Machine

Install [Homebrew](http://brew.sh/):
```
$ cd ~
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
Update Homebrew:

```
$ brew update
```
Before we proceed, we need to update our PATH  in our ~/.bash_profile file to indicate that we want to use Homebrew packages before any system libraries or packages. **This is an absolutely critical step, so be sure not to skip it!**

Open up your ~/.bash_profile  (if it does not exist, create it), and append the following lines:

```
# Homebrew
export PATH=/usr/local/bin:$PATH
```
Then reload the our ~/.bash_profile:

```
$ source ~/.bash_profile
```

And test:

```
$ which python

### OUTPUT MUST BE ###
/usr/local/bin/python
```
First, we’ll use brew to install the required developers tools:

```
$ brew install cmake pkg-config
$ brew install jpeg libpng libtiff openexr
$ brew install eigen tbb
```

We are now ready to set up the Virtual Env

#### iii.Set Up the Virtual Env

Install virtualenv and virtualenvwrapper.These two packages allow us to create separate Python environments for each project we are working on. While installing virtualenv  and virtualenvwrapper is not a requirement to get OpenCV 3.0 and Python 2.7+ up and running on your Ubuntu system, I highly recommend it and the rest of this tutorial will assume you have them installed!

```
### Linux

$ sudo pip install virtualenv virtualenvwrapper 
$ sudo rm -rf ~/.cache/pip

### Macintosh OSX

$ pip install virtualenv virtualenvwrapper
```
Now that we have virtualenv  and virtualenvwrapper  installed, we need to update our ~/.bashrc/ file:
```
### Linux

$ export WORKON_HOME=$HOME/.virtualenvs 

### Linux & Macintosh OSX

$ source /usr/local/bin/virtualenvwrapper.sh
```
This quick update will ensure that both virtualenv  and virtualenvwrapper  are loaded each time you login.
To make the changes to our ~/.bashrc or ~/.bash_profile file take effect, you can either (1) logout and log back in, (2) close your current terminal window and open a new one, or preferably, (3) reload the contents of your ~/.bashrc or ~/.bash_profile file:

```
### Linux 

$ source ~/.bashrc

### Macintosh OSX

$ source ~/.bash_profile
```
Lastly, we can create our cv  virtual environment where we’ll be doing our computer vision development and OpenCV 3.0 + Python 2.7+ installation:
```
### Linux & Macintosh OSX

$ mkvirtualenv cv
```
As I mentioned above, this tutorial covers how to install OpenCV 3.0 and Python 2.7+, so we’ll need to install our Python 2.7 development tools:
```
### Linux

$ sudo apt-get install python2.7-dev
```
Since OpenCV represents images as multi-dimensional NumPy arrays, we better install NumPy into our cv  virtual environment:
```
### Linux & Macintosh OSX

$ pip install numpy
```
### ii.OpenCV
Our environment is now all setup, we can proceed to change to our home directory, pull down OpenCV from GitHub, and checkout the 3.0.0  version:
```
### Linux & Macintosh OSX

$ cd ~ $ git clone https://github.com/Itseez/opencv.git
$ cd opencv 
$ git checkout 3.0.0
```
>You can replace the 3.0.0  version with whatever the current release is. 
>Be sure to check OpenCV.org for information on the latest release.

We also need the opencv_contrib repo as well. Without this repository, we won’t have access to standard keypoint detectors and local invariant descriptors (such as SIFT, SURF, etc.) that were available in the OpenCV 2.4.X version. We’ll also be missing out on some of the newer OpenCV 3.0 features like text detection in natural images:
```
### Linux & Macintosh OSX

$ cd ~ 
$ git clone https://github.com/Itseez/opencv_contrib.git 
$ cd opencv_contrib 
$ git checkout 3.0.0
```
Again, make sure that you checkout the same version for opencv_contrib that you did for opencv above, otherwise you could run into compilation errors.
Time to setup the build:

```
### Linux & Macintosh OSX

$ cd ~/opencv 
$ mkdir build 
$ cd build 

### Linux 

$ cmake -D CMAKE_BUILD_TYPE=RELEASE \ -D CMAKE_INSTALL_PREFIX=/usr/local \ -D INSTALL_C_EXAMPLES=ON \ -D INSTALL_PYTHON_EXAMPLES=ON \ -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \ -D BUILD_EXAMPLES=ON ..

### Macintosh OSX

$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
	-D PYTHON2_PACKAGES_PATH=~/.virtualenvs/cv/lib/python2.7/site-packages \
	-D PYTHON2_LIBRARY=/usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7/bin \
	-D PYTHON2_INCLUDE_DIR=/usr/local/Frameworks/Python.framework/Headers \
	-D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON \
	-D BUILD_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules ..
```

Here some **very important options**, read carefully :

    CMAKE_BUILD_TYPE : This option indicates that we are building a release binary of OpenCV.
    CMAKE_INSTALL_PREFIX : The base directory where OpenCV will be installed.
    PYTHON2_PACKAGES_PATH : The explicit path to where our site-packages  directory lives in our cv  virtual environment.
    PYTHON2_LIBRARY : Path to our Hombrew installation of Python.
    PYTHON2_INCLUDE_DIR : The path to our Python header files for compilation.
    INSTALL_C_EXAMPLES : Indicate that we want to install the C/C++ examples after compilation.
    INSTALL_PYTHON_EXAMPLES : Indicate that we want to install the Python examples after complication.
    BUILD_EXAMPLES : A flag that determines whether or not the included OpenCV examples will be compiled or not.
    OPENCV_EXTRA_MODULES_PATH : This option is extremely important — here we supply the path to the opencv_contrib repo 
    				that we pulled down earlier, indicating that OpenCV should compile the extra modules as well.

>The compiling options changes depending on the version of python and most depending on how we are installing all the libraries and the dependencies.


> In order to build OpenCV 3.1.0, you need to set -D INSTALL_C_EXAMPLES=OFF (rather than ON) in the cmake command. There is a bug in the OpenCV v3.1.0 CMake build script that can cause errors if you leave this switch on. Once you set this switch to off, CMake should run without a problem.

Now we can finally compile OpenCV:
```
### Linux & Macintosh OSX

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
### Linux & Macintosh OSX

$ sudo make install 
$ sudo ldconfig
```
If you’ve reached this step without an error, OpenCV should now be installed in ** /usr/local/lib/python2.7/site-packages**.
However, our cv virtual environment is located in our home directory — thus to use OpenCV within our cv environment, we first need to sym-link OpenCV into the site-packages directory of the cv virtual environment:
```
### Linux 

$ cd ~/.virtualenvs/cv/lib/python2.7/site-packages/ 
$ ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so

### Macintosh OSX

 	
$ cd ~/.virtualenvs/cv/lib/python2.7/site-packages/
$ ls -l cv2.so 
-rwxr-xr-x  1 adrian  staff  2013052 Jun  5 15:20 cv2.so

```
**Congratulations! You have successfully installed OpenCV 3.0 with Python 2.7+ bindings on your Linux/MacintoshOSX system!**
To confirm your installation, simply ensure that you are in the cv virtual environment, followed by importing cv2:
```
### Linux 

$ workon cv 

### Linux & Macintosh OSX

$ python 
```
Your output should be:
```
>>> import cv2
>>> cv2.__version__ '3.0.0'
```

### iii.Tensorflow

For the installation of TensorFlow we can follow the instruction in the Official Turorial, linked at the end, for the Ubuntu case, after activate the envirorment:
```
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl 
```
Test the installation, open a terminal and type the following:
```
$ workon cv 
$ python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```	
Let’s run a Model:
```
$ cd /lib/python2.7/dist-packages/tensorflow/models/image/mnist/
$ workon cv
$ python convolutional.py
```
### iv.CUDA
  
Here is covered only the setup of a the preinstalled CUDA, for an indeep tutorial follow the Official One for TensorfFlow or for MAC OSX (Both linked at the end) :

####Setup GPUs

#####1) Download CUDA 6.5
```
$ cd ~
$ wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/cudnn-6.5-linux-x64-v2.tgz
```
#####2) Add to your .bashrc
```	
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-7.0/lib64:/home/users/N!OMUSUARI/cudnn-6.5-linux-x64-v2"
$ export CUDA_HOME=/usr/local/cuda-7.0
```
  
### v.Other Libraries
 To install other libraries, with the Virtual Environment activated you have just to type:
 ```	
$ pip install packege-name
```
Some usefull, could be:
- [Pillow](https://pypi.python.org/pypi/Pillow);
- [ProgressBar](https://pypi.python.org/pypi/progressbar);
- [Pandas](http://pandas.pydata.org/);

 (I will update the list when i found someone useful)

##4. Conclusions

At the end of the guide we have OpenCV, VirtualEnv and CUDA globally installed on our machine.
Then we have Python and Tensorflow installed and working on the VirtualEnv, in this way we have the global and fixed libraries installed correctly and inside the environment all the python Libraries and Tensorflow to not get in conflict with the global ones.

##5. References
Here below all the usefull link to tutorial, guides that I used:
- [Guide VirtualEnv+OpenCV for Linux](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/);
- [Guide VirtualEnv+OpenCV for MacOSX](http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/);
- [Official Guide to install OpenCV](http://docs.opencv.org/3.0-last-rst/doc/tutorials/introduction/linux_install/linux_install.html);
- [Official Guide to install Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#virtualenv-installation);
- [Official Guide to install CUDA MacOSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html#installation);
- [Official Guide to install CUDA Linux](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz49aYMjbtZ).

##6. Copyright

According to the LICENSE file of the original code,
- Me and original author hold no liability for any damages;
- Do not use this on commercial!.

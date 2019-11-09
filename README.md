[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1275605.svg)](https://doi.org/10.5281/zenodo.1275605)

# A Classification of Vocal Emotions using Convolution Neural Nets.

Details of the baseline CNN set-up is given below. This is for a class project for CS230 in Autumn 2019.

# Panotti: A Convolutional Neural Network classifier for multichannel audio waveforms

<img src="https://upload.wikimedia.org/wikipedia/commons/a/af/Panoteanen.jpg" alt="Panotti image" height="200">
<i>(Image of large-eared Panotti people, Wikipedia)</i><br>


This is a version of the [audio-classifier-keras-cnn](https://github.com/drscotthawley/audio-classifier-keras-cnn) repo (which is a hack of **@keunwoochoi**'s compact_cnn code).  Difference with Panotti is, it has been generalized beyond mono audio, to include stereo or even more "channels."  And it's undergone many refinements.

*NOTE: The  majority of issues people seem to have in using this utility, stem from inconsistencies in their audio datasets. This is to the point where I hesitate to delve into such reports. I suggest trying the binaural audio example and see if your same problems arise.* -SH


## Installation 

### Preface: Requirements
Probably Mac OS X or Linux. (Windows users: I have no experience to offer you.)
Not everything is required, here's a overview:

* Required: 
	* Python 3.5
	* numpy
	* keras
	* tensorflow 
	* librosa
	* matplotlib
	* h5py
* Optional: 
	* sox ("Sound eXchange": command-line utility for examples/binaural. Install via "apt-get install sox")
	* pygame (for exampes/headgames.py)
	* For sorting-hat: flask, kivy kivy-garden
	
...the `requirements.txt` file method is going to try to install both required and optional packages.

### Installation:
`git clone https://github.com/drscotthawley/panotti.git`

`cd panotti`

`pip install -r requirements.txt`


## Demo
I'm not shipping this with any audio but you can generate some for the 'fake binaural' example (requires sox):

    cd examples
    ./binaural_setup.sh
    cd binaural
    ../../preprocess_data.py --dur=2 --clean
    ../../train_network.py


## Quick Start
* Make a folder called `Samples/` and inside it create sub-folders with the names of each category you want to train on. Place your audio files in these sub-folders accordingly. 
* run `python preprocess_data.py`
* run `python train_network.py`
* run `python eval_network.py`  - This applies the trained network to the testing dataset and gives you accuracy reports.



## Data Preparation
### Data organization:
Sound files should go into a directory called `Samples/` that is local off wherever the scripts are being run.  Within `Samples`, you should have subdirectories which divide up the various classes.

Example: for the [IDMT-SMT-Audio-Effects database](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/audio_effects.html), using their monophonic guitar audio clips...

    $ ls -F Samples/
    Chorus/  Distortion/  EQ/  FeedbackDelay/  Flanger/   NoFX/  Overdrive/  Phaser/  Reverb/  SlapbackDelay/
    Tremolo/  Vibrato/
    $
(Within each subdirectory of `Samples`, there are loads of .wav or .mp3 files that correspond to each of those classes.)

*"Is there any sample data that comes with this repo?"*  Not the data itself, but check out the `examples/` directory. ;-)


### Data augmentation & preprocessing:

#### (Optional) Augmentation:

The "augmentation" will [vary the speed, pitch, dynamics, etc.](https://bmcfee.github.io/papers/ismir2015_augmentation.pdf) of the sound files ("data") to try to "bootstrap" some extra data with which to train.  If you want to augment, then you'll run it as

`$ python augment_data.py <N>  Samples/*/*`

where *N* is how many augmented copies of each file you want it to create.  It will place all of these in the Samples/ directory with some kind of "_augX" appended to the filename (where X just counts the number of the augmented data files).
For augmentation it's assumed that all data files have the same length & sample rate.

#### (Required) Preprocessing:
When you preprocess, the data-loading will go *much* faster (e.g., 100 times faster) the next time you try to train the network. So, preprocess.

Preprocessing will pad the files with silence to fit the length to the length of the longest file and the number of channels to the file with the most channels. It will then generate mel-spectrograms of all data files, and create a "new version" of `Samples/` called `Preproc/`.

It will do an 80-20 split of the dataset, so within `Preproc/` will be the subdirectories `Train/` and `Test/`. These will have the same subdirectory names as `Samples/`, but all the .wav and .mp3 files will have ".npy" on the end now.  Datafiles will be randomly assigned to `Train/` or `Test/`, and there they shall remain.

To do the preprocessing you just run

`$ python preprocess_data.py`


## Training & Evaluating the Network
`$ python train_network.py`
That's all you need.  (I should add command-line arguments to adjust the layer size and number of layers...later.)

It will perform an 80-20 split of training vs. testing data, and give you some validation scores along the way.  

It's set to run for 2000 epochs, feel free to shorten that or just ^C out at some point.  It automatically does checkpointing by saving(/loading) the network weights via a new file `weights.hdf5`, so you can interrupt & resume the training if you need to.

After training, more diagnostics -- ROC curves, AUC -- can be obtained by running

`$ python eval_network.py`

*(Changing the `batch_size` variable between training and evaluation may not be a good idea.  It will probably screw up the Batch Normalization...but maybe you'll get luck.)*



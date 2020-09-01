Privacy-preserving resolution of the titanic challenge using CKKS homomorphic encryption
==============================

**Author:** Arthur Pignet

This project implements a solution to the titanic data science problem using homomorphic encryption.
The Titanic problem from Kaggle, [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/), is a well-known classification problem. The objective is to predict, from a set of given features, whether or not someone survived to the famous sinking. 
 
Homomorphic encryption is a type of encryption which allows one to make computations over encrypted data without decrypting it.
Here, we use the CKKS scheme implemented in [TenSEAL](https://github.com/OpenMined/TenSEAL). TenSEAL is an open-source python library, mainly written in C++, and built on top of the Microsoft SEAL library. 

The project can be split in two parts. First we compare three implementations of logistic regression, one from the scikit-learn library (as telltale), and two homemade logistic regressions, one encrypted and the other unencrypted. 
We demonstrate that the use of the CKKS scheme does not impact the model performance, but only the memory and time complexities.
In fact, the same performance as the Scikit-learn implementation can be reached, when one increases the number of epochs of training ( encrypted or unencrypted).
So as to speed up the encrypted training process, a training using multiprocessing is implemented.
Currently, a work around batching the data is in progress.

Then, the second part is a fully-private titanic resolution scenario, with two actors. Bob will stand for the data holder, and Alice for the data scientist. 
Bob will send the encrypted data to Alice, who will train a logistic regression on those. With this trained model, Alice will be able to provide Bob encrypted predictions. From these predictions on both labelled and unlabelled data, Bob will evaluate the model, and thus will decide to use these predictions. He can for instance make a Kaggle submission. 
The submission score is compared to a submission score obtained with Scikit-learn logistic regression.   


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── \_can_be_deleted   <- Trash bin (!! git ignored)
    │
    ├── confidential       <- Confidential documents, data, etc. (!! git ignored)
    │
    ├── data
    │   ├── quick_demo     <- Small subset of the original, immutable data dump, used for quick demo.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    |
    ├── models             <- Trained and serialized models.
    │                         
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   |                     the creator's initials, and a short `-` delimited description.        
    │   ├──0-ap-perf-eval-enc-LR    <- performance evalution of encrypted logistic regression, 
    |   |                               compared to unencrypted logistic regression, and scikit learn logistic regression. 
    │   ├──1-ap-Alice               <- Alice's side. Training a model over encrypted data. The model is then use to predict.
    │   ├──1-ap-Bob                 <- Bob's side. Process and encrypt the data. Sends them to Alice for training and prediction.
    │   ├──Appendix-A-ap-processing <- Data processing and feature engineering.
    │   └──Appendix-B-poly-approxs  <- polynomial approximations of sigmoid and log functions. 
    |    
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    |   ├── quick_demo     <- Small subset of the original raw dataset, used for quick demo.
    │   ├── log            <- Contains the .log files, generated by logging.
    │   └── submission     <- Contains the .csv files with predictions on test set, for kaggle submission 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models             <- Definitions of model classes used in the notebooks 
        │   ├── Alice_LR.py    <-Logistic regression model, used by Alice in the final scenario, notebook 3
        |   ├── encrypted_LR   <-Encrypted homemade logistique regression. Model used in the notebook 2
        │   └── unencrypted_LR <- Unencrypted copy of the encrypted LR. 
        |                         All the operation and implementation ought to be the same as in encrypted LR. 
        |                         Used in the first notebook, for comparison purposes. 
        │
        └── commmunication  <- Class holding the communication protocol. 
            └── Actor.py    <- Class based on the module socket. 
                               Implement a communication protocole between Alice and Bob
     
     
     
## Install everything : 

### Homomorphic encryption : TenSEAL 

Most of the project use the TenSEAL library. **Note that you currently need to build tenseal from source, as we use tenseal's features which have not been released yet.**
Depending on your platform be sure to get the following requirements : 
- **Linux:** A modern version of GNU G++ (>= 6.0) or Clang++ (>= 5.0).
- **MacOS:** Xcode toolchain (>= 9.3)
- **Windows:** Microsoft Visual Studio (>= 10.0.40219.1, Visual Studio 2010 SP1 or later).

You will also need [CMake (3.12 or higher)](https://cmake.org/install/) and [Protocol Buffers](https://developers.google.com/protocol-buffers/docs/downloads) for serialization.
If you are on Windows, you will first need to build SEAL library using Visual Studio. I recommend to check the instructions in the [TenSEAL README](https://github.com/OpenMined/TenSEAL) and [Building Microsoft SEAL](https://github.com/microsoft/SEAL#windows).
 
Once you made sure to have all the requirements installed, you can clone the tenseal repository, get the submodules, and trigger the installation.


```bash
$ git clone https://github.com/OpenMined/TenSEAL.git
$ cd TenSEAL/
$ git submodule init
$ git submodule update
$ pip install .
```

More details to perform such installation (ie build it from source) can be found directly here [TenSEAL](https://github.com/OpenMined/TenSEAL)
### Other dependencies
 
More generally, everything you need can be installed with pip, from the requirement.txt file at the root of the repository: 

```bash
pip install -r requirement.txt
```

## Tested configuration

The project has been ran on a Google Cloud Computing virtual machine, with the following configuration:

- **Machine configuration** : c2-standard-8 (8 vCPUs, Intel Cascade Lake, 32 GB memory)
- **Boot disk** : 100 GB standard persistent disk
- **Image** : Debian GNU/Linux 10 (buster)

Please denote that the project needs lots of memory (I recorded a peak memory of over 27 Gb). 

****This configuration is absolutely not mandatory, it is only an example of a configuration where the project was built and successfully tested.****

--------

All the project was done during a summer internship (June 2020 to August 2020) at [Equancy](http://www.equancy.fr/en). I would like to thank Hervé MIGNOT (Equancy) who drove me during these months. 


<p><small>Project based on the <a target="_blank" href="http://git.equancy.io/tools/cookiecutter-data-science-project/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

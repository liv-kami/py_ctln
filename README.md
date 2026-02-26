# The py_ctln Package README

**Disclaimer**: Please note that this package is in its earliest beta phase. Nothing here is finalized whatsoever. Use at your own risk!

## What is py_ctln?

`py_ctln` is a python package made by the applied mathematician Olivia Kaminske (Liv)
as a way to provide computational tools for use in the study and research of 
Combinatorial Threshold Linear Networks (CTLNs).

CTLNs are a mathematical model of neural networks (like the brain) that provide insight into the dynamics of these 
systems. CTLNs in particular are geared toward a branch of computational neuroscience called **connectomics**. 
Connectomics is the field concerned with understanding the connection between neural connectivity and neural network 
behavior. In other words, we're trying to understand why the particular way in which the neurons in our brains are 
connected ends up creating the sophisticated brain activity we see.

CTLNs were invented by Katherine Morrison and Carina Curto some time ago, and much research has been done on the topic by
them, their students, and the broader mathematics and neuroscience communities. In the course of this research, computational
tools have been invaluable as a way to explore, identify, and prove ideas related to the topic. Thus far, much of this
has been done in MatLab. Unfortunately, MatLab has many drawbacks including the expense, difficulty of learning, lack of 
resources, and so on. Python is a great solution to many of these problems, so Liv took it upon herself to bring the world
of CTLNs to the python stage.

`py_ctln` is the result of this desire, and it is an ongoing project to convert old MatLab code into a distributable python
package, as well as to provide updates, optimizations, and new functionality made possible by the versatility and ease of use
of the python language.

If you're a seasoned studier of CTLNs, we hope this package can serve you well in your research. If you've never before
seen or heard of CTLNs, we hope this package may be a fun and enticing introduction to the topic for those who may feel
less intimidated learning of such a thing through bits of python code. If you're somewhere in between, we hope this package
can serve you well in your pursuits, be they educational, recreational, or (as we often hope) fantastical.

## Installation

To install the latest version of `py_ctln`, simply run the following command in your terminal
```
pip install py_ctln
```

If you'd like to install a specific version, you may do so in the form
```
pip install py_ctln==VERSION_GOES_HERE
```

pip has some other features that can be useful in particular cases, and more information on that can be found on pip's 
sites or through external sources. We typically recommend the simple installation of the latest version for the vast
majority of use cases.

### Installing Pre-Releases

If you're looking to install a pre-release (ie. a version that is incomplete and hosted on a branch other than the main
one), you need only do the following:
1. Download and unzip the source code from github
2. Open a terminal and navigate to the folder that resulted from your unzipping
3. Run the command `hatch build` in that folder, which will create a subdirectory called 'dist'
4. In the 'dist' subdirectory, find the file with the extension '.whl'
5. Copy the absolute path to this file, and you may install it for any python environment of your choosing by running 
the command `pip install path_to_file.whl`

Do keep in mind that pre-releases are often unstable or have broken features. Nonetheless, if you choose to use them and
have difficulty in the course of building/installing them, see hatch's documentation for more assistance.

## Usage

To import the library for usage, include the following line of code with your other imports:

```
from py_ctln import CTLN
```

After importing the library, you may use any of the functions via the form
```
CTLN.function_name(parameters)
```

You can also access sets of known/collected CTLNs that come pre-packaged with the distribution of `py_ctln` by using the
code
```
CTLN.collections.type_of_collection(parameters)
```

### !!! Important Note !!!
Many of the functions in the package rely on a user's defined **adjacency matrix**. Do note that CTLNs use an atypical 
definition of such a matrix, such that the more common definition of an adjacency matrix will need to be transposed to 
be used as a CTLN adjacency matrix (and vice versa to go from a CTLN adjacency matrix to a typical digraph adjacency matrix).

If you encounter issues or unexpected behavior/results, **this is the first place to check**.

#### More Questions?

More in-depth usage information and instructions can be found in the [official documentation](https://github.com/liv-kami/py_ctln/wiki).

## Resources, References, and Background

A list of references will be provided here in the future, both to various papers and publications on the topic of CTLNs
as well as any code snippets, referenced topics, etc. that we may want to call back to.

Beyond references, if you are looking for more information on CTLNs, how they work, and the mathematical foundations at play here,
please see the wiki/docs page [here](https://github.com/liv-kami/py_ctln/wiki) where you can find more in depth explanations and
resources on the topic.

## Acknowledgements

Special thanks to the following people for their contributions to this body of work. This is by no means an complete or
exhaustive list, and we offer our thanks to all CTLN researchers regardless of if they are listed below:

- Carina Curto: Along with Katherine Morrison, the co-inventor of the CTLN model.
- Katherine Morrison: Along with Carina Curto, the co-inventor of the CTLN model.
- Caitlyn Parmelee: A CTLN researcher at Keene State College. Undergraduate advisor of the maintainer and creator of 
this repository and a major help in its creation and maintenance.
- Olivia Kaminske: The creator and maintainer of this repository. A (currently) undergraduate student of Caitlyn 
Parmelee's who took on the task of converting old and creating new code for this project (mostly because she is insane
enough to find the task fun).

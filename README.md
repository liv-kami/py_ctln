# The py_ctln Package README
Please note that this package is in its earliest beta phase. Nothing here is finalized whatsoever. Use at your own risk!
## What is py_ctln?
blah blah blah what is the package and whatnot...

## Installation
blah blah blah

#### Installing Pre-Releases
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

## Docs
Looking for documentation? You can find our docs on the "Wiki" tab of this github repo (also located [here](https://github.com/liv-kami/py_ctln/wiki))

## Resources, References, and Background

A list of references: TBD (need for both code snippets and CTLN papers)

A list of CTLN background information TBD (to be included in the docs)

## Acknowledgements
Special thanks to the following people for their contributions to this body of work. This is by no means an complete or exhaustive list, and we offer our thanks to all CTLN researchers regardless of if they are listed below:

- Carina Curto: Along with Katherine Morrison, the co-inventor of the CTLN model.
- Katherine Morrison: Along with Carina Curto, the co-inventor of the CTLN model.
- Caitlyn Parmelee: A CTLN researcher at Keene State College. Undergraduate advisor of the maintainer and creator of this repository and a major help in its creation and maintenance.
- Olivia Kaminske: The creator and maintainer of this repository. A (currently) undergraduate student of Caitlyn Parmelee's who took on the task of converting old and creating new code for this project.

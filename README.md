# Architecting Data Tools

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ramonpzg/architecting_tools)

## Description

This workshop focuses on building python packages based on statistical models that 
consume data. We will take a plain package setup first and then move on to a newer 
approach that might be of interest to the audience

## Objectives

By the end of the workshop, you will be able to
- create a python package
- publish a python package
- understand how to download and use packages from different places
- understand to version different packages

## Assumptions about the Learner

1. At least a year of experience coding
2. Knows how to use virtual environments
3. Does not mind using jupyter notebooks 😎


## Step-by-Step Setup

#### First Step

Open up your terminal and navigate to a directory of your choosing in your 
computer. Once there, run the following command to get the code for the session.

```sh
 git clone https://github.com/ramonpzg/architecting_tools.git
```

Conversely, you can click on the green `download` button at the top and download all
files to your desired folder/directory. Once you download it, unzip it and move on
to the second step.

#### Second Step

To get all dependencies, packages and everything else that would be useful in this
tutorial, you can recreate the environment by first going into the directory for today.

```sh
cd architecting_tools
```

Then you will need to create an environment with all of the dependencies needed 
for the session by running the following command.

```sh
mamba create -n arch_tools python=3.11
mamba activate arch_tools
pip install -r requirements.txt

## OR

conda create -n arch_tools python=3.11
conda activate arch_tools
pip install -r requirements.txt

## OR

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
#### Third Step

Open up VSCode, a Jupyter Notebook, or Jupyter Lab and you should be ready to go.

```sh
code .

## OR

jupyter lab
```

You're all set. Now navigate to `notebooks` directory and open the `00_lasso.ipynb` notebook.



## Resources

**Tools**
- [setuptools](https://setuptools.pypa.io/en/latest/)
- [pdm](https://pdm-project.org/latest/)
- [poetry](https://python-poetry.org/)
- [hatch](https://hatch.pypa.io/latest/)
- [MLServer](https://mlserver.readthedocs.io/en/latest/)

**Tutorials**
- [Python Packages Book](https://py-pkgs.org/welcome)
- [The Carpentries Packaging Tutorial](https://carpentries-incubator.github.io/python_packaging/instructor/index.html)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/overview/)
- [Orchard Dweller Video](https://www.youtube.com/watch?v=cOFyf0_CDhI&ab_channel=OrchardDweller)

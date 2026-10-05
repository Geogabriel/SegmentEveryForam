\# SegmentEveryForam Installation Guide



This guide explains how to install \*\*SegmentEveryForam\*\* on Windows using Anaconda Prompt.



SegmentEveryForam is a foraminifera-focused image segmentation and morphometric analysis workflow built from Segmenteverygrain and adapted for analysis of foraminiferal microscope images.



\## 1. Prerequisites



Before installing SegmentEveryForam, you will need:



\- Windows

\- Anaconda, Miniconda, or Miniforge

\- Git

\- Internet access during installation

\- Sufficient disk space for the machine-learning dependencies



A GPU is not required to run SegmentEveryForam. Compatible GPU acceleration may improve performance, but the workflow can also run on CPU.



\## 2. Open Anaconda Prompt



Open \*\*Anaconda Prompt\*\* from the Windows Start menu.



You should see something similar to:



```text

(base) C:\\Users\\YourName>

```



All commands below should be entered in Anaconda Prompt.



\## 3. Check Git



Check whether Git is installed:



```bash

git --version

```



If Git is available, its version number will be displayed.



If Git is not installed, you can install it with Conda:



```bash

conda install git

```



\## 4. Clone SegmentEveryForam



Navigate to the directory where you want to install SegmentEveryForam.



For example:



```bash

cd %USERPROFILE%

```



Clone the GitHub repository:



```bash

git clone https://github.com/Geogabriel/SegmentEveryForam.git

```



Then enter the repository:



```bash

cd SegmentEveryForam

```



The cloned repository already contains the `environment.yml` file needed to create the SegmentEveryForam software environment.



You do \*\*not\*\* need to download `environment.yml` separately.



\## 5. Create the Conda Environment



Create the environment using the included `environment.yml`:



```bash

conda env create -f environment.yml

```



This command reads `environment.yml`, creates the SegmentEveryForam Conda environment, and installs the required dependencies.



The initial installation may take several minutes.



You only need to create the environment once.



\## 6. Activate SegmentEveryForam



After installation finishes, activate the environment:



```bash

conda activate segmenteveryforam

```



Your prompt should now begin with:



```text

(segmenteveryforam)

```



For example:



```text

(segmenteveryforam) C:\\Users\\YourName\\SegmentEveryForam>

```



\## 7. Verify the Installation



Check the Python version:



```bash

python --version

```



SegmentEveryForam currently uses Python 3.10.



Test whether the package imports correctly:



```bash

python -c "import segmenteveryforam as sef; print('SegmentEveryForam imported successfully')"

```



A successful installation should print:



```text

SegmentEveryForam imported successfully

```



You can also test major machine-learning dependencies:



```bash

python -c "import torch; import tensorflow as tf; import sam2; print('PyTorch:', torch.\_\_version\_\_); print('TensorFlow:', tf.\_\_version\_\_); print('SAM 2: OK')"

```



TensorFlow may display informational messages when it starts. These messages do not necessarily indicate an installation problem. Look for an actual Python error or traceback.



\## 8. Start JupyterLab



From the SegmentEveryForam repository, with the environment activated, run:



```bash

jupyter lab

```



JupyterLab should open in your web browser.



\## 9. Open the SegmentEveryForam Workflow



In JupyterLab, navigate to:



```text

notebooks/

```



and open:



```text

SegmentEveryForam\_workflow.ipynb

```



Run the notebook from the top downward.



The package is imported in Python using:



```python

import segmenteveryforam as sef

import segmenteveryforam.interactions as sfi

```



\## Quick Start



For users who already have Conda and Git installed, the complete initial setup is:



```bash

cd %USERPROFILE%



git clone https://github.com/Geogabriel/SegmentEveryForam.git



cd SegmentEveryForam



conda env create -f environment.yml



conda activate segmenteveryforam



python -c "import segmenteveryforam as sef; print('SegmentEveryForam ready')"



jupyter lab

```



Then open:



```text

notebooks/SegmentEveryForam\_workflow.ipynb

```



\## Using SegmentEveryForam Again



You do \*\*not\*\* need to recreate the environment each time.



For future sessions, open Anaconda Prompt and run:



```bash

cd %USERPROFILE%\\SegmentEveryForam



conda activate segmenteveryforam



jupyter lab

```



\## Updating SegmentEveryForam



To download newer changes from the GitHub repository:



```bash

cd %USERPROFILE%\\SegmentEveryForam



git pull origin main

```



If `environment.yml` has changed and the environment needs to be updated:



```bash

conda env update -f environment.yml --prune

```



Avoid running `git pull` when you have important uncommitted modifications to repository files.



\## Troubleshooting



\### Conda says the environment already exists



Do not create it again. Activate it:



```bash

conda activate segmenteveryforam

```



\### SegmentEveryForam cannot be imported



First confirm that the correct environment is active:



```bash

conda activate segmenteveryforam

```



Then make sure you are inside the cloned repository:



```bash

cd %USERPROFILE%\\SegmentEveryForam

```



Try the import test again:



```bash

python -c "import segmenteveryforam as sef; print('SegmentEveryForam imported successfully')"

```



\### Git is not recognized



Install Git:



```bash

conda install git

```



\### Jupyter is using the wrong Python environment



Inside a notebook cell, run:



```python

import sys

print(sys.executable)

```



The displayed path should point to the `segmenteveryforam` Conda environment.



\## Repository



SegmentEveryForam is available at:



https://github.com/Geogabriel/SegmentEveryForam


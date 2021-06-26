require numpy pytorch
For anyone that wants to try , first you got to install pip
-1 : download get-pip.py ( from here https://bootstrap.pypa.io/get-pip.py

-2 : move the file here C:\Program Files\Autodesk\Maya2022\bin

-3 : from command prompt go to maya bin dir and run mayapy.exe get-pip.py
this will download pip3.7.exe on script directory
-install numpy
from command prompt go to maya bin dir and run mayapy.exe "C:\Program Files\Autodesk\Maya2022\Python37\Scripts\pip3.7.exe" install numpy
-install pytorch
get pip command form pytorch home page, on my case is 
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
 
from command prompt go to maya bin dir and run mayapy.exe "C:\Program Files\Autodesk\Maya2022\Python37\Scripts\pip37.exe" install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

test
import numpy as np
import torch
x = torch.rand(5, 3)
print(x)

import tensorflow as tf;
print(tf.reduce_sum(tf.random.normal([1000, 1000])))

"""
*** What does this script do? ***
1. Load your trained model.
2. Produce required data for grading.
3. Calculate and dump your score.
Step 3 has been implemented. 
You are required to complete step 1 & 2 in generate_answers()

*** Python Packages ***
You may assume the evaluation environment has the following packages installed:
    - torch
    - numpy
    - sklearn
    - matplotlib

If you wish to install extra packages, please list them in requirements.txt and
place it at the same level of run.py
Read https://pip.pypa.io/en/stable/user_guide/#requirements-files for details.

If your have external packages that cannot be installed via pip, please include
them in your submission and make sure it can be imported correctly. Please note
that we will not handle import problems in this case.

*** Report Problems ***
Please email Ziwei (ziwei.xu@u.nus.edu) if you find problems.

*** Edited: 14:00 Mar-9-2020 ***
"""
import os.path as osp
import argparse
import pickle as pk

import torch
from scores import score_all

from torch.utils import data
from torch import nn
from Generator_cnn import Net1  
from Discriminator_cnn import Net  
import numpy as np
# You can import other modules, including your own modules, above this line.

# Replace A0123456X with your matriculation number
STUDENT_MATRIC_NUM = 'A0076918N'


def load_data(root='.'):
    """
    Load test data.
    Returns:
        X_test: torch.Tensor of shape (|X|, 32, 32, 3), hold-out real samples.
        Z_test: torch.Tensor of shape (|Z|, 256), hold-out test noise samples.
    """
    X_test = torch.load(osp.join(root, 'X_test.pt'))
    Z_test = torch.load(osp.join(root, 'Z_test.pt'))
    return X_test, Z_test


def generate_answers(X_test, Z_test):
    """
    Args:
        Please modify as needed.
    Returns:
        gzs, disc_scores_x, disc_scores_gz: torch.Tensor used by score_all()
        Please see the docstring of score_all() for the required shapes.
    """
    device = 'cuda' #CHANGE THIS LINE TO 'cpu' IF NO GPU TO USE
    #load model
    G = Net1()
    G=torch.load('G_cnn.pth')
  
    D = Net() 
    D=torch.load('D_cnn.pth')
      
    print(X_test.shape, Z_test.shape)
    
    #preprocess of X_test
    X_test = X_test.permute(0,3,1,2)
    X_test =  X_test/255
    X_test=X_test.to(device)
    
    #preprocess of Z_test
    Z_repeat= []
    for i in range(len(Z_test)):
        z_repeat=Z_test[i].repeat(12,1)
        #z_repeat=z_repeat.reshape(32,32,3)
        z_repeat=z_repeat.reshape(3,32,32)
        Z_repeat.append(z_repeat)
    Z_repeat = torch.stack(Z_repeat) 
    Z_repeat =Z_repeat.to(device)
    print(X_test.shape, Z_repeat.shape)
    
    gzs = G(Z_repeat)
    gzs = gzs.reshape(len(gzs), 32,32,3)
    disc_scores_x =  D(X_test)
    disc_scores_gz = D(G(Z_repeat).reshape(len(gzs), 3,32,32))
    print(gzs.shape, disc_scores_x.shape, disc_scores_gz.shape)
    
    return gzs, disc_scores_x, disc_scores_gz
    raise NotImplementedError('Implement this to generate required results.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--root', type=str, default='.')
    p.add_argument('--save_score_to', type=str, default='.')
    args = p.parse_args()

    X_test, Z_test = load_data(args.root)
    
    # Please complete generate_answers()
    gzs, disc_scores_x, disc_scores_gz = generate_answers(X_test, Z_test)
    #################################
    #add the below line to run on gpu
    device = 'cuda'
    X_test=X_test.to(device)
    ################################


    # Do not edit code below this line
    score = score_all(X_test, gzs, disc_scores_x, disc_scores_gz)
    student_info = {**{'MatricNum': STUDENT_MATRIC_NUM}, **score}
    pk.dump(
        obj=student_info, 
        file=open(
            osp.join(args.save_score_to, f'{STUDENT_MATRIC_NUM}.pk'), 'wb')
    )

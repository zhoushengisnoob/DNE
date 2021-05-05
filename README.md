# DNE
Code for paper "Direction-Aware User Recommendation Based on Asymmetric Network Embedding"

## Dataset
We have provided nine directed network dataset including all the datasets used in this paper and some other small datasets for fast evaluation.
* Citeseer(labeled)
* Cora(labeled)
* Cocit(labeled)
* Epinions
* LastFM
* Pubmed(labeled)
* Slashdot
* Twitter
*  Wiki

## How to use
We have provided both the **Tensorflow** and **Pytorch**  implementation of DNE.
The requirements of the running environment is listed in **requirements.txt**.
You can create the environment with anaconda: 

    conda install --yes --file requirements.txt

or virtualenv:

    pip install -r requirements.txt

Then, the code can be run by:

    python main_tf.py (for Tensorflow users)

or 

    python main_torch.py (for Pytorch users)

For the parameters used in the code, see the help of the argparse.
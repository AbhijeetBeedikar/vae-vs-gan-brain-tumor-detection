"""brainad: unsupervised brain-tumour detection in MRI with generative models.

Pipeline overview (each step lives in its own module):

    template.py       build a T2-like MNI brain template + brain mask
    registration.py   affine registration (PyTorch) of any scan onto the template
    preprocessing.py  skull-strip, register, normalise and slice IXI / MSD volumes
    data.py           load the preprocessed slices and build subject-level splits
    models/           DCGAN-style autoencoder, VAE and f-AnoGAN
    train.py          training loops
    anomaly.py        reconstruction -> anomaly map -> predicted tumour mask
    metrics.py        Dice, pixel AUROC/AUPRC, slice AUROC, healthy false-positive rate
    evaluate.py       end-to-end evaluation of one trained model
    visualize.py      figures
"""

__version__ = "0.1.0"

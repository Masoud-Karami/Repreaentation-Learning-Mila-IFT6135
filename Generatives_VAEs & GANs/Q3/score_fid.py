import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]

#%%
def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
        To be implemented by you!
        Code adapted from https://github.com/bioinf-jku/TTUR   
        https://github.com/bioinf-jku?utf8=%E2%9C%93&q=&type=&language=
        
        to use PyTorch instead
        of Tensorflow
    """
    
    """
        Calculates the Frechet Inception Distance (FID) to evalulate GANs
        The FID metric calculates the distance between two distributions of images.
        Typically, we have summary statistics (mean & covariance matrix) of one
        of these distributions, while the 2nd distribution is given by a GAN.
        When run as a stand-alone program, it compares the distribution of
        images that are stored as PNG/JPEG at a specified location with a
        distribution given by summary statistics (in pickle format).
        The FID is calculated by assuming that X_1 and X_2 are the activations of
        the pool_3 layer of the inception net for generated samples and real world
        samples respectively.
    """
    
    #raise NotImplementedError(
    #    "TO BE IMPLEMENTED."
    #    "Part of Assignment 3 Quantitative Evaluations"
    #)
    x = np.asarray(list(islice(1000, sample_feature_iterator)))
    y = np.asarray(list(islice(1000, testset_feature_iterator)))
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)
    MSE = np.dot((mu_x - mu_y),(mu_x - mu_y).T)
    sigma = sigma_x + sigma_y - 2 * scipy.linalg.sqrtm(sigma_x * sigma_y)
    FID = MSE + np.trace(sigma)
    return FID

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)

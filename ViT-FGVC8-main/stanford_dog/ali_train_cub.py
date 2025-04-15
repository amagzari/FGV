from fastai.vision.all import *
from self_supervised.layers import *
import sklearn
from scipy.io import loadmat
from utils.custom_vit import *
from utils.attention import *
from utils.object_crops import *
from utils.part_crops import *
from utils.multi_crop_model import *

datapath = ".............."
filenames = get_image_files(datapath/'images')
train_test_split_df = pd.read_csv(datapath/'train_test_split.txt', delimiter=' ', names=['image_id', 'is_train'])
print(train_test_split_df['is_train'].value_counts())
images_df = pd.read_csv(datapath/'images.txt', delimiter=' ', names=['image_id', 'filename'])
print(images_df.head())

merged_df = images_df.merge(train_test_split_df, on='image_id')
fn2istrain = dict(zip(merged_df['filename'], merged_df['is_train']))



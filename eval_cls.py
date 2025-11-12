import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir
from tqdm import tqdm
from utils import viz_pointcloud, rotate_pointcloud
def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    pred_label = torch.zeros_like(test_label)
    # ------ TO DO: Make Prediction ------
    rotate_degs = [0, 0, 0]
    for idx in tqdm(range(test_data.shape[0])):
        test_rotated = rotate_pointcloud(test_data[idx:idx+1], rotate_degs)
        pred_label[idx] = torch.argmax(model.forward(test_rotated.to(args.device)), dim=1)


    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Print indices where prediction is wrong
    wrong_indices = (pred_label != test_label.data).nonzero(as_tuple=True)[0]
    print("Indices with wrong predictions:", wrong_indices.tolist())

    # Visualize
    test_rotated = rotate_pointcloud(test_data[args.i], rotate_degs)
    viz_pointcloud(test_rotated, "{}/pointcloud_{}.gif".format(args.output_dir, args.exp_name), args.device)
    print("Predicted class for point cloud: {}, Actual class: {}".format(pred_label[args.i], test_label.data[args.i]))
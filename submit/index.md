# Q1

- Report the test accuracy.

The test accuracy was fairly high at 0.9811.

- Visualize a few random test point clouds and mention the predicted classes for each. Also visualize at least 1 failure prediction for each class (chair, vase and lamp),  and provide interpretation in a few sentences.  

Correct predictions:

Chair: 

<image src="cls/pointcloud_0.gif">

Vase:

<image src="cls/pointcloud_700.gif">

Lamp:

<image src="cls/pointcloud_800.gif">

Incorrect predictions:

Ground truth: Chair, predicted: Lamp

<image src="cls/pointcloud_77_wrong.gif">

Ground truth: Vase, predicted: Lamp

<image src="cls/pointcloud_650_wrong.gif">

Ground truth: Lamp, predicted: Vase

<image src="cls/pointcloud_883_wrong.gif">

Interpretations:

Overall, the network distinguishes between the majority of the chair, vase, and lamp objects very well given the accuracy score. The incorrect classifications are likely due to the ambiguous design made in the objects that, lacking further context, is hard to decisively classify as one of the three classes. For example, the chair failure case could be a square-ish design lamp; the vase might look like a lamp on a wooden square stand; and the lamp instance is abstract and lacking features and could be a large vase with a wide body.

# Q2

- Report the test accuracy.

The test accuracy is 0.8028.

- Visualize segmentation results of at least 5 objects (including 2 bad predictions) with corresponding ground truth, report the prediction accuracy for each object, and provide interpretation in a few sentences.

Correct predictions (predicted point cloud | ground truth point cloud):

Accuracy: 0.9711

<image src="seg/pred_34.gif"><image src="seg/gt_34.gif">

Accuracy: 0.9638

<image src="seg/pred_185.gif"><image src="seg/gt_185.gif">

Accuracy: 0.9416

<image src="seg/pred_616.gif"><image src="seg/gt_616.gif">

Incorrect predictions:

Accuracy: 0.3804

<image src="seg/pred_142_bad.gif"><image src="seg/gt_142_bad.gif">

Accuracy: 0.2490

<image src="seg/pred_577_bad.gif"><image src="seg/gt_577_bad.gif">


Interpretations:

While scoring lower accuracy than the classification network, overall the segmentation network performs well on more 'standard' chairs that are upright, four legged, and has an obvious seatback that stands up straight from the seat. However, we can see from the failure cases that it performs poorly on the rarer instances, such as a reclined chair, or a blocky chair with different dimensions than most of the other chairs in the training set.

# Q3

Conduct 2 experiments to analyze the robustness of your learned model. Each experiment is worth 10 points. A maximum of 20 points is possible for this question. Some possible suggestions are:
1. You can rotate the input point clouds by certain degrees and report how much the accuracy falls
2. You can input a different number of points points per object (modify `--num_points` when evaluating models in `eval_cls.py` and `eval_seg.py`)

Please also feel free to try other ways of probing the robustness. 

Deliverables: On your website, for each experiment

- Describe your procedure 
- For each task, report test accuracy and visualization on a few samples, in comparison with your results from Q1 & Q2.
- Provide some interpretation in a few sentences.

## Experiment 1: Rotating point clouds

The input point clouds are all placed at identity rotation poses. Since transformation nets (T-nets) were not implemented for this homework, it is expected that any rotations of the input point cloud will cause accuracy to drop significantly. 

For this experiment, the rotate_degs parameter in the evaluation scripts will be changed so that the original point clouds are all rotated about the origin by this amount. Then, the accuracy will be evaluated and compared to the results from the above questions.

TODO results


## Experiment 2: Reducing input number of sample points per object

Since the networks are trained with 10k points as input, we can see how much the accuracy falls as we provide fewer points in each input point cloud and see how the resulting predictions degrade.

For this experiment, the --num_points parameter will be changed when running the evaluation script, and the resulting accuracies compared to previous results from Q1 and Q2.

TODO results



# Q4 (Bonus)
Not Attempted.
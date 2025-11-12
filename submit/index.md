# Q1


The test accuracy was high at 0.9811.


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


The test accuracy is 0.8028.


Correct predictions (predicted point cloud | ground truth point cloud):

Accuracy for object: 0.9711

<image src="seg/pred_34.gif"><image src="seg/gt_34.gif">

Accuracy for object: 0.9638

<image src="seg/pred_185.gif"><image src="seg/gt_185.gif">

Accuracy for object: 0.9416

<image src="seg/pred_616.gif"><image src="seg/gt_616.gif">

Incorrect predictions:

Accuracy for object: 0.3804

<image src="seg/pred_142_bad.gif"><image src="seg/gt_142_bad.gif">

Accuracy for object: 0.2490

<image src="seg/pred_577_bad.gif"><image src="seg/gt_577_bad.gif">


Interpretations:

While scoring lower accuracy than the classification network, overall the segmentation network performs well on more 'standard' chairs that are upright, four legged, and has an obvious seatback that stands up straight from the seat. However, we can see from the failure cases that it performs poorly on the rarer instances, such as a reclined chair, or a blocky chair with different dimensions than most of the other chairs in the training set.

# Q3

## Experiment 1: Rotating point clouds

The input point clouds are all placed at identity rotation poses. Since transformation nets (T-nets) were not implemented for this homework, it is expected that any significant rotations of the input point cloud will cause accuracy to drop significantly. 

For this experiment, the rotate_degs parameter in the evaluation scripts will be changed so that the original point clouds are all rotated about the origin by this amount. Then, the accuracy will be evaluated and compared to the results from the above questions.

### Classification Model

Here is the obtained accuracy when rotating by the X axis of the object by the following amount (in degrees):

| Rotation (deg) | Accuracy (Classification, overall) | Object GIF | Accuracy (Segmentation, single object) | Segmentation GIF |
|---:|:---:|:---:|:---:|:---:|
| 0 | 0.9811 | <image src="exp_rotate/rot_0.gif"> | 0.9000 | <image src="exp_rotate/rot_0_seg.gif"> |
| 10 | 0.9654 | <image src="exp_rotate/rot_10.gif"> | 0.8995 |  <image src="exp_rotate/rot_10_seg.gif"> |
| 20 | 0.9066 | <image src="exp_rotate/rot_20.gif"> | 0.7200 |  <image src="exp_rotate/rot_20_seg.gif"> |
| 30 | 0.7629 | <image src="exp_rotate/rot_30.gif"> | 0.5600 |  <image src="exp_rotate/rot_30_seg.gif"> |
| 45 | 0.4124* | <image src="exp_rotate/rot_45.gif"> | 0.4600 |  <image src="exp_rotate/rot_45_seg.gif"> |
| 90 | 0.2424* | <image src="exp_rotate/rot_90.gif"> | 0.2600 |  <image src="exp_rotate/rot_90_seg.gif"> |


*: After rotating by more than 30 deg, the shown chair was predicted as a vase. 

Neither network does well when the input object is rotated by any amount. The classification accuracy suffers significantly as soon as the objects are rotated by 10-20 degrees, and is worse than random chance when rotated by close to 90 degrees. This means that the features learned are highly dependent on the chair/vase/lamp being upright and in a canonical rotation. Similarly, the segmentation of the chair parts suffers and starts to bleed into other parts of the chair as it leans more and more forward. This suggests that the current networks are not rotationally invariant at all.


## Experiment 2: Reducing input number of sample points per object

Since the networks are trained with 10k points as input, we can see how much the accuracy falls as we provide fewer points in each input point cloud and see how the resulting predictions degrade.

For this experiment, the --num_points parameter will be changed when running the evaluation script, and the resulting accuracies compared to previous results from Q1 and Q2.


| Number of Points | Accuracy (Classification, overall) | Object GIF | Accuracy (Segmentation, single object) | Segmentation GIF |
|---:|:---:|:---:|:---:|:---:|
| 10000 | 0.9811 | <image src="exp_pts/10k.gif"> | 0.9000 | <image src="exp_pts/10k_seg.gif"> |
| 5000 | 0.9801 | <image src="exp_pts/5k.gif"> | 0.9000 | <image src="exp_pts/5k_seg.gif"> |
| 1000 | 0.9738 | <image src="exp_pts/1k.gif"> | 0.8800 | <image src="exp_pts/1k_seg.gif"> |
| 500 | 0.9685 | <image src="exp_pts/500.gif"> | 0.9000 | <image src="exp_pts/500_seg.gif"> |
| 100 | 0.9381 | <image src="exp_pts/100.gif"> | 0.8800 | <image src="exp_pts/100_seg.gif"> |
| 50 | 0.7827 | <image src="exp_pts/50.gif"> | 0.9200 | <image src="exp_pts/50_seg.gif"> |


The networks perform very well when the number of input points are reduced and randomly sampled from. In the classification case, the chair is still predicted as a chair even when only given 50 points in the point cloud. The accuracy of the classification network doesn't fall much at all until it gets below 100 input points. For the segmentation case, the accuracy of the segmentation output for this object remains similar regardless of the number of input points, suggesting that only 50 points are required to segment this chair generally correctly. We cna conclude that the models trained are robust to different number of input points over a significantly varying range.

# Q4 (Bonus)
Not Attempted.
# Q1

- Report the test accuracy.

The test accuracy was fairly high at 0.9811.

- Visualize a few random test point clouds and mention the predicted classes for each. Also visualize at least 1 failure prediction for each class (chair, vase and lamp),  and provide interpretation in a few sentences.  

Correct predictions:

Incorrect predictions:

Interpretations:

# Q2

- Report the test accuracy.

The test accuracy is 0.8028.

- Visualize segmentation results of at least 5 objects (including 2 bad predictions) with corresponding ground truth, report the prediction accuracy for each object, and provide interpretation in a few sentences.

Correct predictions:

Incorrect predictions:

Interpretations:

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
## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify).
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

[obj]: ./pics/object.png
[passthrough]: ./pics/passthrough.png
[world1]: ./pics/world1.png
[world2]: ./pics/world2.png
[world3]: ./pics/world3.png
[matrix1]: ./pics/matrix1.png
[matrix2]: ./pics/matrix2.png
[matrix3]: ./pics/matrix3.png
[matrix4]: ./pics/matrix4.png
[matrixall]: ./pics/matrixall.png

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

Picture of the cloud after downsizing and passthrough. Leaf size of `0.001` was used. Min, max of `(0.6, 1.1)` were used.
![alt text][passthrough]

Picture after extracting the object from the table. A distance threshold of `0.01` was used.
![alt text][obj]


#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.
In this exercise slightly different config values were used:
```
passthrough.set_filter_limits(0.75, 1.1)
seg.set_distance_threshold(0.02)
ec.set_ClusterTolerance(0.05)
ec.set_MinClusterSize(100)
ec.set_MaxClusterSize(1000)
```

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Using hsv with bin sizes of 32 for the color histogram and the normal histogram, numpy was left to figure out itself.
Also cleaned up both functions by using numpy from the get go instead of converting and looping twice.

Here are the matrix for exercise.

![alt text][matrix1]
![alt text][matrix2]

And for the project

![alt text][matrix3]
![alt text][matrix4]
![alt text][matrixall]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

The settings from exercise 3 were changed a lot at this point and a secondary passthrough over the y axis was also added. Leaf size of `0.005` was used to give the most data as speed was not a huge concern. Passthrough filter in `z` with `(0.6, 0.8)` and `y` with `(-0.55, 0.55)`. The `y` filter was to remove the side of the bins that could be seen to prevent misclassification. Outliers were removed with a std dev of `0.05` and the table was filtered out with a threshold of `0.005`. This low threshold prevented from removing any of the object points. Finally a cluster tolerance of `0.01` was used as some of the objects were very close to each other in world3.

For the training bin sizes of 64 were used for the color space and 100 for the normal space. The model was taught over 30 samples. The same model was used for all 3 worlds. Might have gotten better results if had created a specific model for world2 with only the items in that world.

Currently only the color and normal histogram are used as features. There are some objects that are quite similar in shape (rectangular) which makes differentiating them over the normals very difficult. One thing that could be added would be the actual size of the object. By sorting it becomes orientation independent. This might greatly help for similar shape objects, but of different sizes.
`sort([max(pointsZ)-min(pointsZ), max(pointsY)-min(pointsY), max(pointsX)-min(pointsX)])``

To solve the problem with the world2 misclassification, I would probably look at the color histograms. The two objects are similar in shape, but relatively different in color. By analyzing it a little closer might be able to tweak it sufficiently. Otherwise extra features would need to be added.

World 1 works with 3/3.
![alt text][world1]

World2 the book gets misclassified as biscuits. 4/5
![alt text][world2]

World 3 works with 8/8.
![alt text][world3]

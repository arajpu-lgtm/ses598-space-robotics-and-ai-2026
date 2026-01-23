In this assignment, you will ROS2 and the Turtlesim, to tune a navigator to implement a lawnmower survey (a boustrophedon pattern). The current code will do a pattern shown above. Look at how lawnmower surveys are typically executed, and modify the code as instructed in the descriptions below.
Background
Links to an external site.

Boustrophedon patterns (from Greek: "ox-turning", like an ox drawing a plow) are fundamental survey trajectories used in robotics, space exploration, and Earth observation. These patterns are useful for:

    Space Exploration: Rovers could use boustrophedon patterns to systematically survey areas of interest, ensuring complete coverage when searching for geological samples or mapping terrain. However, due to energy constraints, informative paths are ideally optimized, and this results in paths that are sparser than complete coverage sampling, and yet may still produce high-accuracy reconstructions.

    Earth Observation: Aerial vehicles employ these patterns for:
        Agricultural monitoring and precision farming
        Search and rescue operations
        Environmental mapping and monitoring
        Geological or archaeological surveys

    Ocean Exploration: Autonomous underwater vehicles (AUVs) use boustrophedon patterns to:
        Map the ocean floor
        Search for shipwrecks or aircraft debris
        Monitor marine ecosystems

The efficiency and accuracy of these surveys depend heavily on the robot's ability to follow the prescribed path with minimal deviation (cross-track error). This assignment simulates these real-world challenges in a 2D environment using a first-order dynamical system (the turtlesim robot).
Objective
Links to an external site.

Tune a PD controller to make a first-order system execute the most precise boustrophedon pattern possible. The goal is to minimize the cross-track error while maintaining smooth motion.
Learning Outcomes
Links to an external site.

    Understanding PD control parameters and their effects on first-order systems
    Practical experience with controller tuning
    Analysis of trajectory tracking performance
    ROS2 visualization and debugging

Prerequisites
Links to an external site.
System Requirements
Links to an external site.

Choose one of the following combinations:

    Ubuntu 22.04 + ROS2 Humble
    Ubuntu 23.04 + ROS2 Iron
    Ubuntu 23.10 + ROS2 Iron
    Ubuntu 24.04 + ROS2 Jazzy

Required Packages
Links to an external site.

sudo apt install ros-$ROS_DISTRO-turtlesim
sudo apt install ros-$ROS_DISTRO-rqt*

Python Dependencies
Links to an external site.

pip3 install numpy matplotlib

The Challenge
Links to an external site.
1. Controller Tuning (60 points)
Links to an external site.

Use rqt_reconfigure to tune the following PD controller parameters in real-time:

# Controller parameters to tune
self.Kp_linear = 1.0   # Proportional gain for linear velocity
self.Kd_linear = 0.1   # Derivative gain for linear velocity
self.Kp_angular = 1.0  # Proportional gain for angular velocity
self.Kd_angular = 0.1  # Derivative gain for angular velocity

Performance Metrics:

    Average cross-track error (25 points)
    Maximum cross-track error (15 points)
    Smoothness of motion (10 points)
    Cornering performance (10 points)

2. Pattern Parameters (20 points)
Links to an external site.

Optimize the boustrophedon pattern parameters:

# Pattern parameters to tune
self.spacing = 1.0     # Spacing between lines

    Coverage efficiency (10 points)
    Pattern completeness (10 points)

3. Analysis and Documentation (20 points)
Links to an external site.

Provide a detailed analysis of your tuning process:

    Methodology used for tuning
    Performance plots and metrics
    Challenges encountered and solutions
    Comparison of different parameter sets

Getting Started
Links to an external site.
Repository Setup
Links to an external site.

    Fork the course repository:
        Visit: [https://github.com/DREAMS-lab/RAS-SES-598-Space-Robotics-and-AI](https://github.com/DREAMS-lab/ses598-space-robotics-and-ai-2026/) Links to an external site.
        Click "Fork" in the top-right corner
        Select your GitHub account as the destination

    Clone your fork (outside of ros2_ws):

cd ~/
git clone https://github.com/DREAMS-lab/ses598-space-robotics-and-ai-2026/

    Create a symlink to the assignment in your ROS2 workspace:

cd ~/ros2_ws/src
ln -s ~/ses598-space-robotics-and-ai-2026/assignments/first_order_boustrophedon_navigator .

Building and Running
Links to an external site.

    Build the package:

cd ~/ros2_ws
colcon build --packages-select first_order_boustrophedon_navigator
source install/setup.bash

    Launch the demo:

ros2 launch first_order_boustrophedon_navigator boustrophedon.launch.py

    Monitor performance:

# View cross-track error as a number
ros2 topic echo /cross_track_error

# Or view detailed statistics in the launch terminal

    Visualize trajectory and performance:

ros2 run rqt_plot rqt_plot

Add these topics:

    /turtle1/pose/x
    /turtle1/pose/y
    /turtle1/cmd_vel/linear/x
    /turtle1/cmd_vel/angular/z
    /cross_track_error

Evaluation Criteria
Links to an external site.

    Controller Performance (60%)
        Average cross-track error < 0.2 units (25%)
        Maximum cross-track error < 0.5 units (15%)
        Smooth velocity profiles (10%)
        Clean cornering behavior (10%)

    Pattern Quality (20%)
        Even spacing between lines
        Complete coverage of target area
        Efficient use of space

    Documentation (20%)
        Clear explanation of tuning process
        Well-presented performance metrics
        Thoughtful analysis of results

Submission Requirements
Links to an external site.

    GitHub Repository:
        Commit messages should be descriptive

    Documentation in Repository:
        Update the README.md in your fork with:
            Final parameter values with justification
            Performance metrics and analysis
            Plots showing:
                Cross-track error over time
                Trajectory plot
                Velocity profiles
            Discussion of tuning methodology
            Challenges and solutions

    Submit your work:
        Submit the URL of your GitHub repository
        Ensure your repository is public
        Final commit should be before the deadline

Tips for Success
Links to an external site.

    Start with low gains and increase gradually
    Test one parameter at a time
    Pay attention to both straight-line tracking and cornering
    Use rqt_plot to visualize performance in real-time
    Consider the trade-off between speed and accuracy

Grading Rubric
Links to an external site.

    Perfect tracking (cross-track error < 0.2 units): 100%
    Good tracking (cross-track error < 0.5 units): 90%
    Acceptable tracking (cross-track error < 0.8 units): 80%
    Poor tracking (cross-track error > 0.8 units): 60% or lower

Note: Final grade will also consider documentation quality and analysis depth.
Extra Credit (10 points)
Links to an external site.

Create and implement a custom ROS2 message type to publish detailed performance metrics:

    Define a custom message type with fields for:
        Cross-track error
        Current velocity
        Distance to next waypoint
        Completion percentage
        Other relevant metrics
    Implement the message publisher in your node
    Document the message structure and usage

This will demonstrate understanding of:

    ROS2 message definitions
    Custom interface creation
    Message publishing patterns

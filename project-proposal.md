# computing-at-scale-2025-SB-Final-Project
## Contents  
1. Project Summary
2. Background
3. Implementation
4. Review Process
5. Final Submission
6. References
## Project Summary
The objective of this project is to design, build, and test software in C++ to design spacecraft trajectory for interplanetary transfer for low-thrust spacecraft. The software produced will be similar in main-functionality to NASA's General Mission Analysis Tool (GMAT), but provide a simplified approach to model trajectory. Time-optimal interplanetary transfers will be evaluated using a 'shooting' algorithm and numerical integration to determine the nominal trajectory and a bundle of trajectories perturbed from the nominal trajectory. Additionally, a user of the software will be able to introduce trajectory anomalies into trajectory design to see how these influence the final position of the spacecraft. 
## Background
Time-optimal interplanetary transfers focus on minimizing the time of flight of a spacecraft while satisfying optimality and transversality conditions. To solve for the nominal trajectory, a determinisitc 'shooting' algorithm approach is used to determine the control states of the spacecraft, and subsequently, the trajectory by solving a two-point boundary value problem. Additionally, trajectory deviations should also be generated. To do this, the initial trajectory design must include the expected worst deviations from the pre-designed nominal trajectory. By perturbing the value of the final mass of the spacecraft, a new IVP can be solved to determine these off-nominal trajectories by solving for their control states. To solve for control states, a nonlinear root finder is used based on the minimization of residuals between the currently evaluated control state and optimality/transversality conditions.
## Implementation
To implement this, I plan to build a software suite that includes the residual functions to calculate residuals for the nominal trajectory and bundles, a nonlinear root solver, a numerical integration function to propagate the system forward and backward in time, and a driver program where the user can input start and end conditions. Additionally, I want to create a separate driver program that can be used to modify the trajectory/bundle to see how trajectory anomalies at a certain time influence the final position of the spacecraft.
## Review Process
Using either STK or previously solved transfers, I will use test cases to check the accuracy of my software to see if control states, time of flight, and the trajectories generated match what is expected. 
## Submission
I plan to submit my repository to the Journal of Open Source Software.
## References
1. [Background on Applications/Research](https://www.nature.com/articles/s41598-022-22730-y)
2. [GMAT Software](https://software.nasa.gov/software/GSC-18094-1)   

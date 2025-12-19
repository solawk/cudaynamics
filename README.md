# Software novelty
CUDAynamics is designed for accelerated numerical modeling and analysis of dynamic systems, achieved through parallel computing using CUDA or, with appropriate configuration, OpenMP.

The key difference between CUDAynamics and similar programs is its high interactivity: the user interacts with a single integrated environment, allowing them to select a dynamic system, set its initial conditions and parameters, and visualize the results as interconnected diagrams. The program dynamically responds to changes, instantly updating all related views without the need for relaunching, while calculations, analysis, and visualization are performed within a single application.

# Usage
When the program is launched, three windows open (as shown below): the main window (“CUDAynamics”), the analysis settings window (“Analysis Settings”), and the graph and chart building window (“Graph Builder”).
<p align = "center">
<img width="400" height="500" alt="image" src="https://github.com/user-attachments/assets/c4c6a54c-1828-4eeb-b7e9-0667b475aa6d" />
</p>
In the first half of the main window, one can select a system from the library of systems added to CUDAynamics using the drop-down list. When selecting the Rössler system, the system state variables, simulation step, and system parameters are listed, including the symmetry parameter used in the "VariableSymmetryCD" compositional diagonal modeling method. The drop-down list also allows one to select the numerical integration methods used to derive the finite-difference schemes.

The program's simulation section ("Simulation") displays the memory used by the program for the current configuration and the frame rate. Further down are the simulation settings: trajectory time series and transient time, measured in points or over time, system display type, and playback settings in phase diagrams. There's also a button for returning to the beginning of the current trajectory value buffer, a button for playing back buffers, in which phase diagrams show the system's behavior over time and as a function of parameter value changes, and a button for calculating the next buffer. When varying parameters or system state variables, one can also set their values. The "Compute" button rebuilds the system using a new configuration; it must be pressed to save changes to the initial conditions and system parameters.
	
In the View and Configuration tabs, one can change the view settings (background, font, etc.) and program operation settings.

The Analysis Settings window lists all types of analysis performed in the program
<p align = "center">
<img width="300" height="400" alt="image" src="https://github.com/user-attachments/assets/62aae67f-47fa-4004-916a-f91e2c101975" />
</p>

In this window, one can configure the settings and parameters used in the analysis algorithms. In the Calculate indices section, one can enable or disable the calculation and storage of system characteristics found by this analysis.

In the Graph Builder window, one can select which graph to build and by which parameter, state variable, or characteristic. The Create graph button creates a separate window with the specified graph or diagram.

## Variable Time Series

By selecting the “Variable time series” diagram type for the Rössler system and choosing all three variables we get:
<p align = "center">
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/fdb7b28f-2527-4d79-ba5a-9f0085364f10" />
</p>

## Phase diagram

CUDAynamics has both 2d and 3d phase diagrams, where 2d diagrams are projections of the 3d view drawn against a grid.

Here's an example of 3d phase diagram working for the Rössler system:
<p align = "center">
<img width="500" height="475" alt="image" src="https://github.com/user-attachments/assets/86218622-07ab-4497-a5ca-a2ca31f0c00d" />
<img width="500" height="475" alt="image" src="https://github.com/user-attachments/assets/1ed1fcda-dd6e-4f66-9247-9e36d4b82807" />
</p>
A more comprehensive view of the phase space as seen in the second figure can be achieved by turning on "Use ImPlot3d" in the windows "Plot" options.

If particles mode is turned on in the main window and the "Play" button has been pressed the 3d phase diagram will show the dynamic movement of all calculated variations in phase space through particles as shown below:
<p align = "center">
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/691092e2-2012-4c03-a580-7e8c0b0d663c" />
</p>


## Orbit diagram

To construct single-parameter bifurcation diagrams, select the "Orbit diagram" diagram type in the "Graph Builder" window and choose the system variable to be analyzed. When creating a new window, select a parameter set as varied in the main window from the drop-down list. Shown below are bifurcation diagrams for the z variable for the Rössler system and the system configuration seen earlier with parameter a = 0.2 and parameter b varying in the range [0.05; 0.35], the left diagram show bifurcation by peaks, the right by inter-peak intervals.
<p align = "center">
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/1df1d989-1dbc-4396-a487-1cb29c4b5b8d" />
</p>

Orbit diagrams can also be built in 3d, combining both previous diagrams (although they can be intensive).
<p align = "center">
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/87aad27f-d798-470f-a588-c69f9bc6835d" />
</p>

Orbit diagrams can also be built for specific variations, therefore making a plane section for a specific parameter value in the 3d diagram.
<p align = "center">
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/38e71adb-f1d2-4918-b4aa-9417b9676c15" />
</p>

Orbit diagrams can also be used to build continuation diagrams.
<p align = "center">
<img width="800" height="450" alt="image" src="https://github.com/user-attachments/assets/f0c0ba4f-754c-4aac-ba6a-830e4a80db6a" />
<img width="500" height="425" alt="image" src="https://github.com/user-attachments/assets/460f6106-1a39-4494-84f0-5c36df79fe5f" />
<img width="500" height="425" alt="image" src="https://github.com/user-attachments/assets/e6a03402-4b06-47d2-b124-dabf5b502d11" />
</p>

Orbit diagram windows (as all other diagram windows) have options for diagram output in the "Plot" tab. In the case of orbit diagram point size, shape and color can be changed as can be the type of orbit diagram shown.
<p align = "center">
<img width="500" height="475" alt="image" src="https://github.com/user-attachments/assets/f9b84996-9080-4fc5-b717-ffff45b8ac72" />
</p>

## Heatmap
The Heatmap can be used to make diagrams of relativity of the value of a systems characteristic (indeces, gotten in the result of analysis). 

For efficient and high resolution diagrams the "Hi-Res Mode" option should be checked in the Heatmap Diagram window. The interface will change color when this mode is active.
<p align = "center">
<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/5c63aefc-a2af-4ad3-b0c0-9256bfbf5274" />
</p>

By holding RMB while in some diagrams we can zoom in to a particular section of the diagram, but while holding SHIFT+RMB we can not only zoom in, but set bounds for a new diagram.
<p align = "center">
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/82f4f191-9bd4-4bf1-a466-735689a87276" />
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/4a901583-c9b0-467c-96da-6f6a2982d384" />
</p>

The heatmap diagram also has a crosshair that selects a specific variation
<p align = "center">
<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/33d1ab90-5895-4315-a9da-4dcd683eaedd" />
<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/d591aa91-f5d5-4392-b54e-d53d134c6e70" />
</p>

The Heatmap diagram can also be used for coloring in particles and trajectories in the phase diagrams. This can be done by choosing the appropriate option in the "Colors" tab of the heatmap window.
<p align = "center">
<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/cc7fd6e1-92a7-4bf1-b9f0-f3672fcae3d0" />
</p>

The heatmap plot settings are as shown:
<p align = "center">
<img width="500" height="480" alt="image" src="https://github.com/user-attachments/assets/f7d75bae-3d2d-4038-ae20-c83611755b50" />
</p>

## RGB Heatmap
RGB Heatmap shows three index values for each variation, not one. Although it doesn't have all the funcionality as does the normal heatmap it can be used to make something such as an assessment of basins of attracton for a system.

The following example for this is shown on the "Josephson JMCRL-shunted junction" system.
<p align = "center">
<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/471491c7-239a-4045-9823-67497c23fc66" />

<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/b3ba8599-45cc-4462-a7ea-82decb3f34e6" />
</p>

## Indeces diagram
When selecting this diagram, add the indices to be analyzed and, in the window that appears, select the parameter varied in the main window from the drop-down list. Shown below a diagram of the dependence of the Largest Lyapunov Exponent ("LLE") and periodicity ("Period") for the parameter b for the system configuration shown above and parameter a = 0.2 (the multiple Y - axes can be turned off in the "Plot" submenu).
<p align = "center">
<img width="600" height="450" alt="image" src="https://github.com/user-attachments/assets/ffe004fa-90bf-48c7-8722-ac040f35b931" />
</p>

## Indeces time series
Indeces time series shows the dynamics of a systems indeces. To be built the program needs to go through buffers, so to draw new point the user needs to either manually press the "Next buffer" button or turn on buffer playback via the "Play" button.

The following diagram was achieved for the "Josephson JMCRL-shunted junction" system with the configuration as seen above.
<p align = "center">
<img width="550" height="400" alt="image" src="https://github.com/user-attachments/assets/641cfe2e-9315-421c-b017-e6d3aba2ead9" />
</p>
As seen after a transient chaotic system the systems solution becomes stable.

## Decay Plot
Decay plot can be used to analyze the dynamics of multiple variations of a system. This plot reurns how many variations have an index value less/more then the specified value. Using Orbit diagram to build continuation diagrams or heatmap periodicity diagram, we can see that for the chosen configuration (as seen in configuration in RGB heatmap) the "Josephson JMCRL-shunted junction" system has two attractors in the secified space one with period of 2 and the second with 4.

In the "Decay Plot" window, one can set a second limit value by clicking the "+" button. In the resulting drop-down list, enter the limit values in descending order; in this case, 5 and 3 are specified, so 1 more than periodicity values for attractors. To display the values in the diagram, the program needs to read new buffers, so for the diagram to work, one must manually click the "Next buffer" button or enable buffer playback mode by clicking the "Play" button.
<p align = "center">
<img width="600" height="425" alt="image" src="https://github.com/user-attachments/assets/5cb156bb-6de8-4255-8a21-2d7a7ef628c9" />
</p>
As can be seen, over time the system's solutions converge to periodic regimes.


# Credits
## Developed by 
Alexander Khanov, Ivan Guitor, Nikita Belyaev, Ksenia Shinkar, Maxim Gozhan, Anastasia Karpenko

## Directed by 
Valerii Ostrovskii

## Dependencies
Dear ImGUi by Omar Cornut

Implot by Evan Pezent

CUDA Toolkit 12.6.2 by NVIDIA

OpenMP 2.0 standard

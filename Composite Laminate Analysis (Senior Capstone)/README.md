The following was developed to fulfill the requirements of SE 143A & B --> which is the senior capstone sequence at the University of California San Diego's Aerospace structures program (under the structural engineering department).

The goal of the Capstone was to develop a fully functional and modular code to analyze the structural and dynamic components of a composite VTOL UAV. Each member of our team worked to reach this goal. My job as the composite material lead - was to develop a script which calculates FIRST ply failure in a given laminate at every user discretized node and cross section along the upper and lower skin of each UAV wing. The png above depicts the analyzed cross section and nodes at which analysis was done.


The 'FPF2020' code shown above, is the manifestation of this goal and takes in a variety of inputs as a result. 

In Summary:

1. Via the layup input file, the user is to define laminate ID's for all possible laminates along the airfoil. Those with a background in composite materials know that: each ply of a laminate carries with it its own subsequent material properties, ply thicknesses and rotation angle. The code accounts for this, by pulling the material properties of each ply (for each laminate) from the material database ~ after which material properties for each of the laminate(s) are calculated from ABD matrices.


2. The code needs also an input of in plane forces resultants (NX),(NXY),(NY) for every node in every cross section - as a 3D matrix (each matrix represents a cross section, each row of a matrix represents a node). The code also needs the user to define which laminates ID's correspond to each of these in plane forces.


3. The code then calculates the first failure ply, for each laminate at each point in each cross section. The output of this code comes as a {cell} variable called STORE. Within store there are n cells, corresponding to n cross sections at which an analysis was performed. Within each of these n cells, there is a table containing every node at which a first ply failure analysis was done, the ply which failed, the material of that ply, the number of that ply, the mode of failure, and the in plane force resultants which caused it to fail.


Please see the USER GUIDE, page 24 onwards, for more.

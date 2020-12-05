{\rtf1\ansi\ansicpg1252\cocoartf1671
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 The following was developed to fulfill the requirements of SE 143A & B --> which is the senior capstone sequence at the University of California San Diego's Aerospace structures program (under the structural engineering department).\
\
\
The goal of the Capstone was to develop a fully functional and modular code to analyze the structural and dynamic components of a composite VTOL UAV. Each member of our team worked to reach this goal. My job as the composite material lead - was to develop a script which calculates FIRST ply failure in a given lamina at every user discretized point and cross section, along the upper and lower skin of each UAV wing. The png above depicts the analyzed cross section and nodes at which analysis was done.\
\
The 'FPF2020' code shown above, is the manifestation of this goal and takes in a variety of inputs as a result. Via the layup input file, the user is to define to laminate ID for all possible laminates along the airfoil. Those with a background in composite materials know that, each ply of a laminate carries with it its own subsequent material properties, ply thicknesses and rotation angle. The first part of the code accounts for this, by pulling the material properties of each ply (for each laminate) from the material database ~ after which material properties for each of the laminate(s) are calculated. Then, given an input of stresses NX(axial force), NY, }
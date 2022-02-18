# waveFEniCS
Repository for the UK National GPU Hackathon 2022

## Problem statement

The target of our solver is for high-intensity focused ultrasound 
(HIFU) application. HIFU application is typically model using the linear 
second order wave equation. Together with the source and absorbing boundary 
condition, the problem statement reads

<p align="center">
    <img src=problem_statement.png/>
</p>

To use the finite element method, the problem statement is formulated into 
its variational form that yields

<p align="center">
    <img src=variational_form.png/>
</p>

Each of the terms on the right-hand-side of the variational form represents 
an operator.

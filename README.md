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
an operator. In matrix form, the equation becomes

<p align="center">
    <img src=matrix_form.png/>
</p>

In term of the temporal algorithm, we will use the Runge-Kutta method.

## Matrix-Free Operator 
<p align="center">
    <img src=operators.png/>
</p>

![Selection_063](https://user-images.githubusercontent.com/15614155/155312723-e27da569-2173-4657-8bc6-8fd147c1e01c.png)


### Example Kernel - k>>m~n
![MicrosoftTeams-image](https://user-images.githubusercontent.com/15614155/155314168-9b45db59-a1d9-4d49-b6a4-1c22abc59170.png)



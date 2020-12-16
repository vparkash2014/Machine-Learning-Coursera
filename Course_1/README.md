# Assignments for the first course of this Specialisation.

Below provided a summary of what each assignment was and a summary of Machine Learning algorithm used to solve the problem

Assignment 1: 

Design a simple algorithm to distinguish cat images from non-cat images.

I built a Logistic Regression, using a Neural Network mindset. 

<img src="images/LogReg_kiank.png" style="width:650px;height:400px;">

**Mathematical expression of the algorithm**:
For one example <img src="https://render.githubusercontent.com/render/math?math=x^{(i)}">:

<img src="https://render.githubusercontent.com/render/math?math=z^{(i)} = w^T x^{(i)} + b \tag{1}">

<img src="https://render.githubusercontent.com/render/math?math=\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}">

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}">


The cost is then computed by summing over all training examples:

<img src="https://render.githubusercontent.com/render/math?math=J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}">


**The main steps for building a Neural Network are:**
1. Define the model structure (such as number of input features) 
2. Initialize the model's parameters
3. Loop:
    - Calculate current loss (forward propagation)
        - You get X
        - You compute <img src="https://render.githubusercontent.com/render/math?math=A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})">
        - You calculate the cost function: <img src="https://render.githubusercontent.com/render/math?math=J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})">
        - Here are the two formulas you will be using: 
        
          <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}">


          <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}">


    - Calculate current gradient (backward propagation)
    - Update parameters (gradient descent)
    
*Files*

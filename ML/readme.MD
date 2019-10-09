***This page we will realize some maching learning algorithm from scrath.***

# Chapter 4: Experiments
- We sample a unifrom distribution from multi-binominal distribution by Gibbs sampling.
- run `./hands_on_handbooks_from_scratch/Chp_4_monte_carlo.py`
```
3.1416019193534255
Whether the `pi` we computed is closed to math.pi?: True
```
- reverse sampling for exponent distribution:
![reverse sampling](https://github.com/fooSynaptic/Py_utils/blob/master/ML/srcs/images/chapter4_exp_distribution_by_inverse_sampling.png)
- reject samplning for exponent distribution:
![reject sampling](https://github.com/fooSynaptic/Py_utils/blob/master/ML/srcs/images/chapter4_exp_distribution_reject_sampling.png)


- reference [AI算法工程师手册](http://www.huaxiaozhuan.com/)


***Machine learning from scratch***
- [Linear regression with parameter learn by Gradient descent.](linearRegression.py)
res (5 epoches):
```bash
True W: [22.5, -43.3]
Predict W: [array([ 21.46418781, -42.19876115])]
```
*You may see the gradient may be pos or negative, but they all converged to zeros at last.*
![Loss and gradient upadte process](https://github.com/fooSynaptic/Py_utils/blob/master/images/linearRegression_GradientDescent.png)


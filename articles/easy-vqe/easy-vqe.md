One of the leading algorithms in the NISQ quantum computer is called VQE (Variational Quantum Eigensolver). The name sounds hard, but the explanations are often overwhelmingly difficult[^1]. I would like to approach this with a short approach.

[^1]: For example, https://learn.qiskit.org/course/ch-applications/simulating-molecules-using-vqe

## Conclusion

VQE is, in essence, the following:

```python
qc = QuantumCircuit(1)
qc.ry(Parameter("θ"), 0)
print(VQE(Estimator(), qc, SPSA()).compute_minimum_eigenvalue(Pauli("Z")).optimal_parameters)
```

> {Parameter(θ): 3.141592653589793}

## The detail

The VQE story is generally accompanied by the following difficult explanations.

- Quantum chemical calculations
    - Full CI
    - Quantum phase estimation
- Schrödinger equation and Hamiltonian
- Second quantization
- Eigenvalues of the ground state of the Hamiltonian

However, these are mainly related to how to formulate the problem and how to make mathematical models. Therefore, they can be separated from the main routine of VQE.

The main routine of VQE is essentially the same as the optimization routine of deep learning, which is a numerical calculation to find the minimum value of a given cost function.

The code I wrote at the beginning of this post is too rough, but if we write it more carefully, it would be as follows:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SPSA


qc = QuantumCircuit(1)
qc.ry(Parameter("θ"), 0)

estimator = Estimator()  # a too to define the cost function
operator = Pauli("Z")  # the objective to analyze
optimizer = SPSA()  # an optimizer

vqe = VQE(estimator, qc, optimizer)  # a trainer which trains a model inside a loop
result = vqe.compute_minimum_eigenvalue(operator)

print(result.optimal_parameters)
```

Mathematically, we take a function

{% katex %}
\begin{align*}
f(\theta) &= (1 \quad 0) \begin{pmatrix}
  \cos(\frac{\theta}{2}) & \sin(\frac{\theta}{2}) \\
  - \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
\end{pmatrix}
\begin{pmatrix}
  1 & 0 \\
  0 & -1
\end{pmatrix}
\begin{pmatrix}
  \cos(\frac{\theta}{2}) & -\sin(\frac{\theta}{2}) \\
  \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
\end{pmatrix}
\begin{pmatrix}
1 \\
0
\end{pmatrix} \\
&= (1 \quad 0) \begin{pmatrix}
  \cos(\frac{\theta}{2}) & \sin(\frac{\theta}{2}) \\
  - \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
\end{pmatrix}
\begin{pmatrix}
  \cos(\frac{\theta}{2}) \\
  -\sin(\frac{\theta}{2})
\end{pmatrix} \\
&= \cos \theta
\end{align*}
{% endkatex %}

as the cost function, and {% katex inline %}\theta{% endkatex %} is to be optimized so that {% katex inline %}f(\theta){% endkatex %} is minimized. It is obvious that {% katex inline %}\theta = \pm \pi{% endkatex %} are optima and take the minimum value {% katex inline %}-1{% endkatex %}. Thus, going back to the beginning, we have

> {Parameter(θ): 3.141592653589793}

It is quick to start the story where the cost function {% katex inline %}f(\theta) = \cos \theta{% endkatex %} is given.

- Determine randomly the initial value of {% katex inline %}\theta{% endkatex %}
- Update {% katex inline %}\theta{% endkatex %} iteratively with an appropriate optimizer
- If the loss function value doesn't seem to change, optimization is complete

These procedures are like what we have seen often somewhere... Yes! Deep Learning.

SPSA (Simultaneous Perturbation Stochastic Approximation) is used as an optimizer that fits the situation of NISQ quantum computer, but if we assume an ideal situation where a reliable gradient values can be calculated, there is no problem to optimize with Adam. **It is completely an optimization method for deep learning**.

## Wrap up

I have tried to reduce difficult portions as much as possible to concentrate on the main routine of the VQE, although it becomes a little more complicated than I expected.

There are various types of quantum machine learning, but the above is in essence **optimizing a cost function** created from a quantum mechanical point of view with **an optimization method similar to that often used in deep learning**.

While the case of quantum chemical computation is a direct quantum mechanical problem, problems such as the traveling salesman problem and other route optimization problems are not quantum mechanical in nature. However, even for such problems, in some cases, the (mathematical) formulation of the problem can be done within the framework of quantum computation. **If you find merit in quantum computation**, you can use the formulation in the style of quantum computation and obtain a numerical solution by optimization similar to that used in deep learning.

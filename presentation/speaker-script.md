# Speaker Script

This file contains the exact spoken script for the `presentation/slides.md` deck, organized slide by slide.

## Slide 1: Title

"Hello everyone. Today I will present our project on regularization in neural networks, with a comparison between classical regularization methods and synaptic neural balance. The presentation is based on the report, and I will focus on the motivation, the method, the experimental setup, and the main empirical result."

## Slide 2: Why This Project?

"Neural networks are powerful, but that same flexibility can also make them overfit or train in unstable ways. Because of that, regularization is one of the central tools in deep learning. Most people are already familiar with methods like weight decay and dropout, but synaptic neural balance is less standard and is therefore interesting to test empirically. The main question of this project is simple: can neural balancing outperform standard regularization methods in a controlled experimental setting?"

## Slide 3: What We Compare

"In this project, we compare several representative regularization approaches. These are L2 weight decay, L1 penalty, dropout, early stopping, synaptic neural balance with an L1 cost, and synaptic neural balance with an L2 cost. We evaluate them along three dimensions: generalization, convergence speed, and stability across random seeds. The goal is not to exhaust every possible regularizer, but to compare a practical set of baselines against balancing under the same protocol."

## Slide 4: What Is Synaptic Neural Balance?

"At a high level, synaptic neural balance is a re-scaling operation applied around a hidden unit. We scale the incoming weights by a factor lambda, and the outgoing weights by one over lambda. Under the usual homogeneity assumptions on the activation, this transformation preserves the function represented by the network. That means the input-output mapping stays the same, but the distribution of weight magnitudes changes. The idea is to choose lambda in a way that reduces a chosen weight cost, so balancing can improve the parameterization without changing the function."

## Slide 5: Experimental Setup

"For the controlled experiment in the report, we use MNIST and a small fully connected network with architecture 784 to 256 to 10. Training uses SGD with learning rate 0.001, batch size 256, and a maximum budget of 150 epochs. We evaluate with the fixed seed list 42, 43, and 44. The important point here is fairness: all methods use the same backbone and training protocol unless the regularization method itself is the thing being varied."

## Slide 6: Hyperparameter Selection

"Hyperparameters are selected using a validation split only, and the test set is reserved for final reporting. To tune the methods, we use Gaussian-Process Bayesian Optimization. The reason is that training models is expensive, so Bayesian Optimization is more sample-efficient than a dense grid search. Each tunable method receives the same search budget: 12 total BO trials, made up of 4 random initial points and 8 guided points. In this setup, one BO step means evaluating one candidate configuration across the fixed seed list."

## Slide 7: Main Results on MNIST

"This is the main results slide. We report both final test accuracy and the number of epochs needed to first reach 90 percent test accuracy. The standout result is synaptic balance with the L1 cost. It reaches 90 percent in only 7.7 epochs and finishes at 96.32 percent final test accuracy. For comparison, the best non-balance baseline is dropout, which reaches the threshold at 70 epochs and finishes at 92.48 percent. Synaptic balance with the L2 cost also performs well, reaching 92.75 percent with faster convergence than the standard baselines."

## Slide 8: Key Takeaways

"The main takeaway is that synaptic balance with L1 is the strongest method in this controlled experiment. It is not only the best in final accuracy, but also much faster to converge. That combination matters, because it suggests balancing may help both optimization and generalization. A second takeaway is that the choice of balancing cost matters: the L1 and L2 versions do not behave the same, and the L1 version is much stronger in this setting."

## Slide 9: Interpretation and Caveats

"Our interpretation is that balancing may be improving the training dynamics and the final generalization performance at the same time. At the same time, we should be careful not to overclaim. These results come from one dataset and one relatively small architecture. In addition, Bayesian Optimization itself still involves design choices such as the search bounds, acquisition strategy, and total budget. There is also one reporting limitation in the current table: the plain baseline has only one completed seed, so its uncertainty estimate is incomplete."

## Slide 10: What This Suggests

"The broader implication is that neural balancing looks like a promising complement to classical regularization rather than just an isolated trick. It would be valuable to test it on larger networks, more datasets, data-scarce regimes, and combinations with standard regularizers. More generally, this project suggests that regularization is not only about adding penalties such as L1 or L2. Re-parameterizing the network while preserving its function can also have a strong practical effect."

## Slide 11: Conclusion

"To conclude, we compared standard regularizers with synaptic neural balance in a controlled MNIST experiment. In this setting, synaptic balance with an L1 cost performs best, and it is especially notable for both fast convergence and higher final test accuracy. Thank you, and I am happy to take questions."

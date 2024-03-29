# Intro to sparse data and embeddings
I used Tensorflow to make some word embeddings.

## View my work
My code can be found here: [intro_to_sparse_data_and_embeddings.md](intro_to_sparse_data_and_embeddings.md). I created the markdown version of the `.ipynb` notebook because `.ipynb` files often don't render in Github.



## Summary
* The model performs sentiment analysis on a movie review dataset (positive vs. negative).
* The words are first represented as one-hot vectors from a limited vocabulary of 50 terms.
* A simple logistic regression is applied (`tf.estimator.LinearClassifier`) (test AUC: 0.87036055), then a feed-forward neural net (`tf.estimator.DNNClassifier`) (test AUC: 0.8653846).
* The sparse features are combined into an embedding layer of dimension 2, which is lower dimensional than the indicator columns, and the models re-run with different hyperparameters.

How does the 2d embedding get assigned?
> That is, the model learns the best way to map your input numeric categorical values to the embeddings vector value in order to solve your problem.

From [here](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)

## Results
| Hidden units | batch norm | steps | optimizer | learning rate | test AUC |
| -- | -- | -- | -- | -- | -- |
| [20,20] | False | 1000 | adagrad | 0.1 | 0.8666485 |
| [20,20] | False | 100 | adagrad | 0.1 | 0.71556914 |
| [40,20] | False | 1000 | adagrad | 0.1 | 0.86758274 |
| [40,20] | False | 5000 | adagrad | 0.1 | 0.8707501 |
| [40,20] | False | 10000 | adagrad | 0.1 | 0.8708241 |
| [40,20] | False | 10000 | adagrad | 0.08 | 0.8709081 |
| [40,20] | True | 10000 | adagrad | 0.08 | __0.8709133__ |
| [40,20] | NA | 1000 | adam | 0.001 | 0.7138001 |
| [40,20] | NA | 1000 | adam | 0.0001 | 0.5597429 |
| [40,20] | NA | 1000 | adam | 0.01 | 0.86710733 |

I did not get into regularization on the optimizer.

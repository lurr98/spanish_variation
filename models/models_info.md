## Support Vector Machines

[Available on scikit-learn](https://scikit-learn.org/stable/modules/svm.html)

### Hypotheses

- SVMs could prove to be a good model if the features described by Lipski are actually present in the data and there are enough features to construct a separating hyperplane. Looking at it from the other point of view, if the SVM performs well, it means that *the features described by Lipski are actually present in the data and it is possible to construct an appropriate hyperplane from them*.
- could size be a problem? SVM is good with little training data.

### Advantages:

- Effective in high dimensional spaces.
- Still effective in cases where number of dimensions is greater than the number of samples.
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient. :fire:
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
    - Can help determine whether data is linearly separable. :fire:
- Determining feature importance is possible, there is a designated attribute (`coef_`) for linear kernels but even for the non-linear there is a workaround. :fire:

### Disadvantages

- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation. :skull:


## Decision Trees

[Available on scikit-learn](https://scikit-learn.org/stable/modules/tree.html)

[(Random Forest)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### Hypotheses

- Decision trees / Random forest can deal with irregular patterns and therefore non-smooth decision boundaries, which NNs apparantly struggle with ([see this medium entry](https://medium.com/geekculture/why-tree-based-models-beat-deep-learning-on-tabular-data-fcad692b1456)). Since it is very likely that the patterns found in the data are pretty irregular for the task of dialect classification, DTs/RF could actually have an advantage over BERT models.

### Advantages

- Simple to understand and to interpret. Trees can be visualized. :fire:
- Requires little data preparation. Other techniques often require data normalization, dummy variables need to be created and blank values to be removed. Some tree and algorithm combinations support missing values. :fire:
- The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree. :fire:
- Able to handle both numerical and categorical data. However, the scikit-learn implementation does not support categorical variables for now. Other techniques are usually specialized in analyzing datasets that have only one type of variable. See algorithms for more information.
- Able to handle multi-output problems.
- Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret. :fire:
- Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model. :fire:
- Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.


### Disadvantages

- Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble. :skull:
- Predictions of decision trees are neither smooth nor continuous, but piecewise constant approximations as seen in the above figure. Therefore, they are not good at extrapolation. :skull:
- The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.
- There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
- :warning: Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.


## BERT Model

[Spanish BERT model](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)

### Hypotheses

- Many of the features described by Lipski are restricted to vernacular speech, which means that they may not be present in the corpus at all. Transformer models are very good at picking up on subtle cues (due to next token/sentence prediction training objective?) that may be present in the corpus, so they could help alleviate this issue.

### Advantages

- SOTA
- Results by Bernier-Colborne et al. (2019) in the CLI shared task seem to indicate that recent developments in contextual embedding representations may also yield performance improvement in language identification applied to similar languages, varieties, and dialects. :fire:


### Disadvantages

- Black-box model, not explainable. :skull:
- High time and computational cost. :skull:
- :warning: BERT could simply make use of topics to determine class, e.g. texts from Panama in the corpus will probably talk about Panama more often than other texts


## How to Incorporate Features

### What type of features?

- character ngrams or token ngrams?
    - high performing approaches have used "word-based" representations (so unigrams?) and character ngrams of higher order (4-, 5-, 6-grams)
    - when to cut off?
        - scikit-learn has the *min_df* parameter for *CountVectorizer* which is essentially a cut-off based on **document frequency**
    - include stop words?
- the selected linguistically tailored features

### Pipeline for linear models

1. preprocessing
    - read corpus
    - tokenisation, lemmatisation and POS-tagging
    - lowercase everything
        - since capitalisation only happens for NE in Spanish and these are already tagged anyway, so lowercasing **all** data should not pose a problem
2. extract features
    - character ngrams
        - 4-grams
        - 5-grams
        - 6-grams
    - word ngrams
        - unigrams
    - linguistically tailored features :white_check_mark:
3. split data in train, dev, test
    - maybe balance data?
        - undersample?
            - the **SVM** class has an attribute named *class_weight* which can be set to *balanced*
            - :warning: this cannot balance out the different average paragraph length though!
4. combine features
    - concatenate
5. train model on data 
    - SVM 
    - Decision Tree / Random Forest
    - experiment with different features
        - feature engineering
        - ablation studies
# Little overview over the evaluation

## Different feature types

For the linear models, I used in total **seven** different feature configurations, namely:

- *tailored features*
- *counts*: ngram features (based on absolute counts, no stopwords)
- *counts_nofeat*: ngram features (based on absolute counts, tailored feature overlap as stopwords)
- *tf*: ngram features (based on relative term frequency, no stopwords)
- *tf_nofeat*: ngram features (based on relative term frequency, tailored feature overlap as stopwords)
- *both_counts_nofeat*: tailored features + ngram features (based on absolute counts, tailored feature overlap as stopwords)
- *both_tf_nofeat*: tailored features + ngram features (based on relative term frequency, tailored feature overlap as stopwords)

For the transformer models, I simply use the preprocessed text and let it be truncated and padded by the tokeniser.


## Evaluation of tailored features

1. very bad performance across all models
    - accuracy around 0.07 for SVM and DT, around 0.08 for RF
    - macro F1 around 0.05 for SVM, 0.069 for DT and 0.076 for RF
2. confusion matrices show some learning effect though
    - for some countries, the right class is assigned more often than each wrong class individually
    - SVM focuses more on ES, while DT and RF focus more on BO and DO
    - all three models focus a lot on PY, why?
3. feature weights indicate high focus on VOSEO-related features
    - but for DT and RF, clitics seem to be important too
    - SUBJINV and MASNEG are also relevant for all models
    - MUYISIMO is relevant for SVM
4. feature weights by class need a thorough examination and interpretation
    - **BUT:** we can already see some results that support the literature
        - e.g. MASNEG has the highest feature importance for CU (SVM model)
    - generally a big focus on VOSEO


# ngram features (tf)

1. very little difference between *tf* and *tf_nofeat*
    - counter-intuitive bc *tf_nofeat* has most pronouns as stopwords
        - but maybe the declension is still a cue?
        - maybe the ngram model does not rely on VOSEO too much?
    - for DT, *tf* vs. *tf_nofeat* changes on which class the model "focuses" on
2. pretty good performance for SVM model
    - 0.64 accuracy and 0.65 macro F1 (*tf* and *tf_nofeat*)
    - also, the model doesn't confuse the countries as could be expected
3. not so great performance from the DT 
    - 0.38 accuracy and 0.45 macro F1 (*tf* and *tf_nofeat*)
    - focuses on ES or MX (why? the data is balanced)
4. feature importances for the whole model show that the ngram models mostly focus on the countrys' names, nationalities and landmarks such as "montevideo" (both SVM and DT)

### Accuracy

#### Standard Experiments

|                            | SVM  | DT   | Transformer |
|----------------------------|------|------|-------------|
| tailored                   | 0.08 | 0.07 |             |
| ngrams_tf                  | 0.64 | 0.38 |             |
| ngrams_tf_nofeat           | 0.64 | 0.38 |             |
| ngrams_counts              | 0.63 | 0.39 |             |
| ngrams_counts_nofeat       | 0.63 | 0.39 |             |
| both_tf_nofeat             | 0.61 | 0.38 |             |
| both_counts_nofeat         | 0.63 | 0.39 |             |
| on text                    |      |      |     0.67    |


#### Ablation Study: GROUP

|                            | SVM  | DT   | Transformer |
|----------------------------|------|------|-------------|
| tailored_grouped           | 0.15 | 0.14 |             |
| ngrams_tf_nofeat_grouped   | 0.66 | 0.41 |             |
| both_tf_nofeat_grouped     | 0.64 | 0.41 |             |
| on text                    |      |      |     0.8     |


#### Ablation Study: NONES

|                            | SVM  | DT   | Transformer |
|----------------------------|------|------|-------------|
| ngrams_counts_nofeat_nones | 0.43 | 0.16 |             |
| ngrams_tf_nofeat_nones     | 0.55 | 0.16 |             |
| both_counts_nofeat_nones   | 0.45 | 0.17 |             |
| both_tf_nofeat_nones       | 0.51 | 0.17 |             |
| on text                    |      |      |     0.35    |


### MACRO F1

#### Standard Experiments

|                            | SVM   | DT   |
|----------------------------|-------|------|
| tailored                   | 0.05  | 0.07 |
| ngrams_tf                  | 0.65  | 0.45 |
| ngrams_tf_nofeat           | 0.65  | 0.45 |
| ngrams_counts              | 0.64  | 0.45 |
| ngrams_counts_nofeat       | 0.64  | 0.45 |
| both_tf_nofeat             | 0.6   | 0.45 |
| both_counts_nofeat         | 0.64  | 0.45 |


#### Ablation Study: GROUP

|                            | SVM   | DT   |
|----------------------------|-------|------|
| tailored_grouped           | 0.12  | 0.13 |
| ngrams_tf_nofeat_grouped   | 0.66  | 0.44 |
| both_tf_nofeat_grouped     | 0.64  | 0.44 |


#### Ablation Study: NONES

|                            | SVM   | DT   |
|----------------------------|-------|------|
| ngrams_counts_nofeat_nones | 0.44  | 0.18 |
| ngrams_tf_nofeat_nones     | 0.54  | 0.17 |
| both_counts_nofeat_nones   | 0.45  | 0.18 |
| both_tf_nofeat_nones       | 0.51  | 0.17 |


### grouped

classes are not balanced anymore!!!
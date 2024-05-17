# tailored features

1. very bad performance across all models
    - accuracy around 0.07 for all models
    - macro F1 around 0.05 for SVM, 0.065 for DT and 0.07 for RF
2. confusion matrices show some learning effect though
    - for some countries, the right class is assigned more often than each wrong class individually
    - SVM focuses more on ES, while DT and RF focus more on BO and DO
    - all three models focus a lot on PY, why?
3. feature weights indicate high focus on voseo related features
    - but for DT and RF, clitics seem to be important too
    - SUBJINV and MASNEG are also relevant for all models
    - MUYISIMO is relevant for SVM


# ngram features (tf)

1. very little difference between *tf* and *tf_nofeat*
    - counter-intuitive bc *tf_nofeat* has most pronouns as stopwords
        - but maybe the declension is still a cue?
        - maybe the ngram model does not rely on voseo too much?
2. pretty good performance for SVM model (I think)
    - 0.64 accuracy and 0.65 macro F1 (*tf* and *tf_nofeat*)
    - also, the model doesn't confuse the countries as could be expected
3. not so great performance from the DT 
    - 0.38 accuracy and 0.45 macro F1 (*tf* and *tf_nofeat*)
    - focuses on ES or MX (why? the data is balanced)

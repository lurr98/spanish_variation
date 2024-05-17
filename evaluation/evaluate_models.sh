python3 evaluation_pipeline.py SVM_models/SVM_model_both_tf_nofeat_2024-05-15 both_features/tf_nofeat/both_features_spmatrix_tf_nofeat_dev evaluation/SVM_models/evaluation_SVM_model_both_tf_nofeat.txt -ev f1 accuracy confusion_matrix class_report > out.txt
rm -f out.txt
# python3 evaluation_pipeline.py DT_models/DT_model_both_tf_nofeat_2024-05-15 both_features/tf_nofeat/both_features_spmatrix_tf_nofeat_dev evaluation/DT_models/evaluation_DT_model_both_tf_nofeat.txt -ev f1 accuracy confusion_matrix class_report > out.txt
# rm -f out.txt
# python3 evaluation_pipeline.py DT_models/DT_model_ngrams_tf_2024-05-15 ngram_features/tf/ngram_frequencies_spmatrix_tf_dev evaluation/DT_models/evaluation_DT_model_tf.txt -ev f1 accuracy confusion_matrix class_report -grid > out.txt
# rm -f out.txt

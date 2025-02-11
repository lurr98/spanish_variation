touch out.txt

##############
# EVALUATION #
##############

# SVM
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_ngrams_tf_nofeat_2024-05-23 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_test evaluation/SVM_models/evaluation_SVM_model_ngrams_tf_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_ngrams_counts_nofeat_2024-05-27 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_test evaluation/SVM_models/evaluation_SVM_model_ngrams_counts_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_tailored__2024-08-06 tailored_features/tailored_features_tf_test evaluation/SVM_models/evaluation_SVM_model_tailored_tf_test.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_tailored_tf_2024-08-13 tailored_features/tailored_features_tf_dev evaluation/SVM_models/evaluation_SVM_model_tailored_tf_dev_updated.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_tailored_tf_2024-08-13 tailored_features/tailored_features_tf_test evaluation/SVM_models/final_evs/evaluation_SVM_model_tailored_tf_test.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_tailored_counts_2024-08-14 tailored_features/tailored_features_counts_test evaluation/SVM_models/final_evs/evaluation_SVM_model_tailored_counts_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_tailored__2024-08-14 tailored_features/tailored_features_counts_dev evaluation/SVM_models/final_evs/evaluation_SVM_model_tailored_counts_dev.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_both_counts_nofeat_2024-08-14 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_test evaluation/SVM_models/final_evs/evaluation_SVM_model_both_counts_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_both_tf_nofeat_2024-08-14 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_test evaluation/SVM_models/final_evs/evaluation_SVM_model_both_tf_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt

# grouped
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_tailored_tf_grouped_2024-08-13 tailored_features/tailored_features_tf_grouped_test evaluation/SVM_models/final_evs/evaluation_SVM_model_tailored_tf_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_tailored_counts_grouped_2024-08-14 tailored_features/tailored_features_counts_grouped_test evaluation/SVM_models/final_evs/evaluation_SVM_model_tailored_counts_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_ngrams_tf_nofeat_grouped_2024-06-06 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_test evaluation/SVM_models/GROUP/evaluation_SVM_model_ngrams_tf_nofeat_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_ngram_counts_nofeat_grouped_2024-07-26 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_grouped_test evaluation/SVM_models/GROUP/evaluation_SVM_model_ngrams_counts_nofeat_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_both_tf_nofeat_grouped_2024-08-14 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_test evaluation/SVM_models/final_evs/evaluation_SVM_model_both_tf_nofeat_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_both_tf_nofeat_tf_grouped_2024-08-04 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_tf_grouped_test evaluation/SVM_models/GROUP/evaluation_SVM_model_both_tf_nofeat_tailored_tf_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_both_counts_nofeat_grouped_2024-08-14 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_grouped_test evaluation/SVM_models/final_evs/evaluation_SVM_model_both_counts_nofeat_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt

# nones
# python3 -u evaluation_pipeline.py linear SVM_models/NONES/SVM_model_ngrams_counts_nofeat_nones_2024-06-27 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_test evaluation/SVM_models/evaluation_SVM_model_ngrams_counts_nofeat_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/NONES/SVM_model_both_counts_nofeat_nones_2024-08-14 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_test evaluation/SVM_models/final_evs/evaluation_SVM_model_both_counts_nofeat_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/NONES/SVM_model_ngrams_tf_nofeat_nones_2024-06-02 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_test evaluation/SVM_models/evaluation_SVM_model_ngrams_tf_nofeat_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/NONES/SVM_model_both_tf_nofeat_nones_2024-08-14 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_test evaluation/SVM_models/final_evs/evaluation_SVM_model_both_tf_nofeat_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt


# DT
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_ngrams_tf_nofeat_2024-05-22 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_test evaluation/DT_models/evaluation_DT_model_ngrams_tf_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_ngrams_counts_nofeat_2024-05-24 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_test evaluation/DT_models/evaluation_DT_model_ngrams_counts_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_tailored_tf_2024-08-13 tailored_features/tailored_features_tf_dev evaluation/DT_models/evaluation_DT_model_tailored_tf_dev_updated.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_tailored_tf_2024-08-13 tailored_features/tailored_features_tf_test evaluation/DT_models/final_evs/evaluation_DT_model_tailored_tf_test.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
python3 -u evaluation_pipeline.py linear DT_models/DT_model_tailored_counts_2024-08-14 tailored_features/tailored_features_counts_dev evaluation/DT_models/final_evs/evaluation_DT_model_tailored_counts_dev.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_both_counts_nofeat_2024-08-14 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_test evaluation/DT_models/final_evs/evaluation_DT_model_both_counts_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_both_tf_nofeat_2024-08-14 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_test evaluation/DT_models/final_evs/evaluation_DT_model_both_tf_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt

# grouped
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_tailored_tf_grouped_2024-08-13 tailored_features/tailored_features_tf_grouped_test evaluation/DT_models/final_evs/evaluation_DT_model_tailored_tf_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_tailored_counts_grouped_2024-08-14 tailored_features/tailored_features_counts_grouped_test evaluation/DT_models/final_evs/evaluation_DT_model_tailored_counts_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_ngrams_tf_nofeat_grouped_2024-06-06 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_test evaluation/DT_models/GROUP/evaluation_DT_model_ngrams_tf_nofeat_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_ngram_counts_nofeat_grouped_2024-07-26 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_grouped_test evaluation/DT_models/GROUP/evaluation_DT_model_ngrams_counts_nofeat_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_both_tf_nofeat_grouped_2024-08-14 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_test evaluation/DT_models/final_evs/evaluation_DT_model_both_tf_nofeat_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_both_counts_nofeat_grouped_2024-08-14 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_grouped_test evaluation/DT_models/final_evs/evaluation_DT_model_both_counts_nofeat_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt

# nones
# python3 -u evaluation_pipeline.py linear DT_models/NONES/DT_model_ngrams_counts_nofeat_nones_2024-05-27 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_test evaluation/DT_models/evaluation_DT_model_ngrams_counts_nofeat_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/NONES/DT_model_both_counts_nofeat_nones_2024-08-14 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_test evaluation/DT_models/final_evs/evaluation_DT_model_both_counts_nofeat_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/NONES/DT_model_ngrams_tf_nofeat_nones_2024-06-05 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_test evaluation/DT_models/evaluation_DT_model_ngrams_tf_nofeat_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/NONES/DT_model_both_tf_nofeat_nones_2024-08-14 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_test evaluation/DT_models/final_evs/evaluation_DT_model_both_tf_nofeat_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt


# TRANSFORMER
# python3 -u evaluation_pipeline.py transformer transformer_models/transformer_model_5_epochs_feat none evaluation/transformer_models/evaluation_transformer_models_5_epochs_feat.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py transformer transformer_models/transformer_model_5_epochs_nofeat none evaluation/transformer_models/evaluation_transformer_models_5_epochs_nofeat_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# ython3 -u evaluation_pipeline.py transformer transformer_models/transformer_model_5_epochs_grouped none evaluation/transformer_models/evaluation_transformer_model_5_epochs_grouped_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# ython3 -u evaluation_pipeline.py transformer transformer_models/transformer_model_5_epochs_nones none evaluation/transformer_models/evaluation_transformer_model_5_epochs_nones_test.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt


######################
# FEATURE IMPORTANCE #
######################

# SVM
# python3 -u interpretation_features.py SVM_models/SVM_model_ngrams_tf_nofeat_2024-05-23 svm nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_test >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_ngrams_counts_nofeat_2024-05-27 svm nofeat -ftp ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_test >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_tailored_tf_2024-08-13 svm tailored -ftp tailored_features/tailored_features_tf_test
# python3 -u interpretation_features.py SVM_models/SVM_model_tailored_counts_2024-08-14 svm tailored -ftp tailored_features/tailored_features_counts_test
# python3 -u interpretation_features.py SVM_models/SVM_model_both_counts_nofeat_2024-08-14 svm both -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_test >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_both_tf_nofeat_2024-08-14 svm both -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_tf_test >> out.txt

# grouped
# python3 -u interpretation_features.py SVM_models/GROUP/SVM_model_tailored_tf_grouped_2024-08-13 svm tailored -ftp tailored_features/tailored_features_tf_grouped_test >> out.txt
# python3 -u interpretation_features.py SVM_models/GROUP/SVM_model_tailored_counts_grouped_2024-08-14 svm tailored -ftp tailored_features/tailored_features_counts_grouped_test >> out.txt
# python3 -u interpretation_features.py SVM_models/GROUP/SVM_model_ngrams_tf_nofeat_grouped_2024-06-06_grouped svm nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_test >> out.txt
# python3 -u interpretation_features.py SVM_models/GROUP/SVM_model_both_tf_nofeat_grouped_2024-08-14 svm both -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_test >> out.txt
# python3 -u interpretation_features.py SVM_models/GROUP/SVM_model_both_counts_nofeat_grouped_2024-08-14 svm both -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_grouped_test >> out.txt

# nones
# python3 -u interpretation_features.py SVM_models/NONES/SVM_model_ngrams_counts_nofeat_nones_2024-06-27 svm nones -ftp ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_test >> out.txt
# python3 -u interpretation_features.py SVM_models/NONES/SVM_model_both_counts_nofeat_nones_2024-08-14 svm bothn -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_test >> out.txt
# python3 -u interpretation_features.py SVM_models/NONES/SVM_model_ngrams_tf_nofeat_nones_2024-06-02 svm nones -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_test >> out.txt
# python3 -u interpretation_features.py SVM_models/NONES/SVM_model_both_tf_nofeat_nones_2024-08-14 svm bothn -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_test >> out.txt


# DT
# python3 -u interpretation_features.py DT_models/DT_model_ngrams_tf_nofeat_2024-05-22 dt nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_test >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_ngrams_counts_nofeat_2024-05-24 dt nofeat -ftp ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_test >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_tailored_tf_2024-08-13 dt tailored -ftp tailored_features/tailored_features_tf_test >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_tailored_counts_2024-08-14 dt tailored -ftp tailored_features/tailored_features_counts_test >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_both_counts_nofeat_2024-08-14 dt both -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_test >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_both_tf_nofeat_2024-08-14 dt both -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_test >> out.txt

# grouped
# python3 -u interpretation_features.py DT_models/GROUP/DT_model_tailored_tf_grouped_2024-08-13 dt tailored -ftp tailored_features/tailored_features_tf_grouped_test >> out.txt
# python3 -u interpretation_features.py DT_models/GROUP/DT_model_tailored_counts_grouped_2024-08-14 dt tailored -ftp tailored_features/tailored_features_counts_grouped_test >> out.txt
# python3 -u interpretation_features.py DT_models/GROUP/DT_model_ngrams_tf_nofeat_grouped_2024-06-06_grouped dt nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_test >> out.txt
# python3 -u interpretation_features.py DT_models/GROUP/DT_model_both_tf_nofeat_grouped_2024-08-14 dt both -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_test >> out.txt
# python3 -u interpretation_features.py DT_models/GROUP/DT_model_both_counts_nofeat_grouped_2024-08-14 dt both -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_grouped_test >> out.txt

# nones
# python3 -u interpretation_features.py DT_models/NONES/DT_model_ngrams_counts_nofeat_nones_2024-05-27 dt nones -ftp ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_test >> out.txt
python3 -u interpretation_features.py DT_models/NONES/DT_model_both_counts_nofeat_nones_2024-08-14 dt bothn -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_test >> out.txt
# python3 -u interpretation_features.py DT_models/NONES/DT_model_ngrams_tf_nofeat_nones_2024-06-05 dt nones -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_test >> out.txt
# python3 -u interpretation_features.py DT_models/NONES/DT_model_both_tf_nofeat_nones_2024-08-14 dt bothn -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofea_nones_test >> out.txt
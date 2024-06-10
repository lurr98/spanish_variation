touch out.txt

##############
# EVALUATION #
##############

# # SVM
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_ngrams_tf_2024-05-23 ngram_features/tf/ngram_frequencies_spmatrix_tf_dev evaluation/SVM_models/evaluation_SVM_model_ngrams_tf.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_ngrams_tf_nofeat_2024-05-23 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev evaluation/SVM_models/evaluation_SVM_model_ngrams_tf_nofeat.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_ngrams_counts_2024-05-28 ngram_features/counts/ngram_frequencies_spmatrix_counts_dev evaluation/SVM_models/evaluation_SVM_model_ngrams_counts.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_ngrams_counts_nofeat_2024-05-27 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_dev evaluation/SVM_models/evaluation_SVM_model_ngrams_counts_nofeat.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_tailored__2024-05-22 tailored_features/tailored_features_dev evaluation/SVM_models/evaluation_SVM_model_tailored.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_both_counts_nofeat_2024-05-27 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_dev evaluation/SVM_models/evaluation_SVM_model_both_counts_nofeat.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_both_tf_nofeat_2024-05-27 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev evaluation/SVM_models/evaluation_SVM_model_both_tf_nofeat.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# # grouped
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_tailored__2024-06-06_grouped tailored_features/tailored_features_grouped_dev evaluation/SVM_models/GROUP/evaluation_SVM_model_tailored_grouped.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_ngrams_tf_nofeat_grouped_2024-06-06_grouped ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_dev evaluation/SVM_models/GROUP/evaluation_SVM_model_ngrams_tf_nofeat_grouped.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/GROUP/SVM_model_both_tf_nofeat_grouped_2024-06-06_grouped both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_dev evaluation/SVM_models/GROUP/evaluation_SVM_model_both_tf_nofeat_grouped.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# # nones
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_ngrams_counts_nofeat_nones_2024-05-27 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_dev evaluation/SVM_models/evaluation_SVM_model_ngrams_counts_nofeat_nones.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/SVM_model_both_counts_nofeat_nones_2024-05-28 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_dev evaluation/SVM_models/evaluation_SVM_model_both_counts_nofeat_nones.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/NONES/SVM_model_ngrams_tf_nofeat_nones_2024-06-02 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_dev evaluation/SVM_models/evaluation_SVM_model_ngrams_tf_nofeat_nones.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear SVM_models/NONES/SVM_model_both_tf_nofeat_nones_2024-06-05 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_dev evaluation/SVM_models/evaluation_SVM_model_both_tf_nofeat_nones.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# 
# 
# # DT
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_ngrams_tf_2024-05-22 ngram_features/tf/ngram_frequencies_spmatrix_tf_dev evaluation/DT_models/evaluation_DT_model_ngrams_tf.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_ngrams_tf_nofeat_2024-05-22 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev evaluation/DT_models/evaluation_DT_model_ngrams_tf_nofeat.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_ngrams_counts_2024-05-28 ngram_features/counts/ngram_frequencies_spmatrix_counts_dev evaluation/DT_models/evaluation_DT_model_ngrams_counts.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_ngrams_counts_nofeat_2024-05-24 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_dev evaluation/DT_models/evaluation_DT_model_ngrams_counts_nofeat.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_tailored__2024-05-21 tailored_features/tailored_features_dev evaluation/DT_models/evaluation_DT_model_tailored.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_both_counts_nofeat_2024-05-23 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_dev evaluation/DT_models/evaluation_DT_model_both_counts_nofeat.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_both_tf_nofeat_2024-05-25 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev evaluation/DT_models/evaluation_DT_model_both_tf_nofeat.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# # grouped
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_tailored__2024-06-06_grouped tailored_features/tailored_features_grouped_dev evaluation/DT_models/GROUP/evaluation_DT_model_tailored_grouped.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_ngrams_tf_nofeat_grouped_2024-06-06_grouped ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_dev evaluation/DT_models/GROUP/evaluation_DT_model_ngrams_tf_nofeat_grouped.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/GROUP/DT_model_both_tf_nofeat_grouped_2024-06-06_grouped both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_dev evaluation/DT_models/GROUP/evaluation_DT_model_both_tf_nofeat_grouped.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# # nones
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_ngrams_counts_nofeat_nones_2024-05-27 ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_dev evaluation/DT_models/evaluation_DT_model_ngrams_counts_nofeat_nones.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/DT_model_both_counts_nofeat_nones_2024-05-28 both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_dev evaluation/DT_models/evaluation_DT_model_both_counts_nofeat_nones.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/NONES/DT_model_ngrams_tf_nofeat_nones_2024-06-05 ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_dev evaluation/DT_models/evaluation_DT_model_ngrams_tf_nofeat_nones.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py linear DT_models/NONES/DT_model_both_tf_nofeat_nones_2024-06-05 both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_dev evaluation/DT_models/evaluation_DT_model_both_tf_nofeat_nones.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# 
# 
# # RF
# python3 -u evaluation_pipeline.py linear RF_models/RF_model_tailored__2024-05-22 tailored_features/tailored_features_dev evaluation/RF_models/evaluation_RF_model_tailored.txt -ev f1 accuracy confusion_matrix class_report -grid -save_pred >> out.txt
# # grouped
# python3 -u evaluation_pipeline.py linear RF_models/RF_model_tailored__2024-05-26_grouped tailored_features/tailored_features_dev evaluation/RF_models/evaluation_RF_model_tailored_grouped.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt


# # TRANSFORMER
# python3 -u evaluation_pipeline.py transformer transformer_models/transformer_model_5_epochs none evaluation/transformer_models/evaluation_transformer_model_5_epochs.txt -ev f1 accuracy confusion_matrix class_report -save_pred >> out.txt
# python3 -u evaluation_pipeline.py transformer transformer_models/transformer_model_5_epochs_grouped none evaluation/transformer_models/evaluation_transformer_model_5_epochs_grouped.txt -ev f1 accuracy confusion_matrix class_report >> out.txt


######################
# FEATURE IMPORTANCE #
######################

# SVM
# python3 -u interpretation_features.py SVM_models/SVM_model_ngrams_tf_2024-05-23 svm ngrams -ftp ngram_features/tf/ngram_frequencies_spmatrix_tf_dev >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_ngrams_tf_nofeat_2024-05-23 svm nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_ngrams_counts_2024-05-28 svm ngrams -ftp ngram_features/counts/ngram_frequencies_spmatrix_counts_dev >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_ngrams_counts_nofeat_2024-05-27 svm nofeat -ftp ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_dev >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_tailored__2024-05-22 svm tailored -ftp tailored_features/tailored_features_dev >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_both_counts_nofeat_2024-05-27 svm both -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_dev >> out.txt
# python3 -u interpretation_features.py SVM_models/SVM_model_both_tf_nofeat_2024-05-27 svm both -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev >> out.txt
# # grouped
python3 -u interpretation_features.py SVM_models/GROUP/SVM_model_tailored__2024-06-06_grouped svm tailored -ftp tailored_features/tailored_features_dev >> out.txt
python3 -u interpretation_features.py SVM_models/GROUP/SVM_model_ngrams_tf_nofeat_grouped_2024-06-06_grouped svm nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_dev >> out.txt
python3 -u interpretation_features.py SVM_models/GROUP/SVM_model_both_tf_nofeat_grouped_2024-06-06_grouped svm both -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_dev >> out.txt
# # nones
python3 -u interpretation_features.py SVM_models/NONES/SVM_model_ngrams_counts_nofeat_nones_2024-05-27 svm nones -ftp ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_dev >> out.txt
python3 -u interpretation_features.py SVM_models/NONES/SVM_model_both_counts_nofeat_nones_2024-05-28 svm bothn -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_dev >> out.txt
python3 -u interpretation_features.py SVM_models/NONES/SVM_model_ngrams_tf_nofeat_nones_2024-06-02 svm nones -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_dev >> out.txt
python3 -u interpretation_features.py SVM_models/NONES/SVM_model_both_tf_nofeat_nones_2024-06-05 svm bothn -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_dev >> out.txt




# DT
# python3 -u interpretation_features.py DT_models/DT_model_ngrams_tf_2024-05-22 dt ngrams -ftp ngram_features/tf/ngram_frequencies_spmatrix_tf_dev >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_ngrams_tf_nofeat_2024-05-22 dt nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_ngrams_counts_2024-05-28 dt ngrams -ftp ngram_features/counts/ngram_frequencies_spmatrix_counts_dev >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_ngrams_counts_nofeat_2024-05-24 dt nofeat -ftp ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_dev >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_tailored__2024-05-21 dt tailored -ftp tailored_features/tailored_features_dev >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_both_counts_nofeat_2024-05-23 dt both -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_dev >> out.txt
# python3 -u interpretation_features.py DT_models/DT_model_both_tf_nofeat_2024-05-25 dt both -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev >> out.txt
# # grouped
python3 -u interpretation_features.py DT_models/GROUP/DT_model_tailored__2024-06-06_grouped dt tailored -ftp tailored_features/tailored_features_dev >> out.txt
python3 -u interpretation_features.py DT_models/GROUP/DT_model_ngrams_tf_nofeat_grouped_2024-06-06_grouped dt nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_dev >> out.txt
python3 -u interpretation_features.py DT_models/GROUP/DT_model_both_tf_nofeat_grouped_2024-06-06_grouped dt both -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_grouped_dev >> out.txt
# # nones
python3 -u interpretation_features.py DT_models/NONES/DT_model_ngrams_counts_nofeat_nones_2024-05-27 dt nones -ftp ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_dev >> out.txt
python3 -u interpretation_features.py DT_models/NONES/DT_model_both_counts_nofeat_nones_2024-05-28 dt bothn -ftp both_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones_dev >> out.txt
python3 -u interpretation_features.py DT_models/NONES/DT_model_ngrams_tf_nofeat_nones_2024-06-05 dt nones -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_dev >> out.txt
python3 -u interpretation_features.py DT_models/NONES/DT_model_both_tf_nofeat_nones_2024-06-05 dt bothn -ftp both_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones_dev >> out.txt

# RF
# python3 -u interpretation_features.py RF_models/RF_model_tailored__2024-05-22 rf tailored -ftp tailored_features/tailored_features_dev >> out.txt
# grouped
# python3 -u interpretation_features.py RF_models/RF_model_tailored__2024-05-26_grouped rf tailored -ftp tailored_features/tailored_features_dev >> out.txt


# python3 -u interpretation_features.py SVM_models/SVM_model_tailored__2024-05-22 svm tailored -ftp tailored_features/tailored_features_dev >> out.txt
# # python3 -u interpretation_features.py DT_models/DT_model_tailored__2024-05-21 dt tailored -ftp tailored_features/tailored_features_dev >> out.txt
# # python3 -u interpretation_features.py RF_models/RF_model_tailored__2024-05-22 rf tailored -ftp tailored_features/tailored_features_dev >> out.txt
# # python3 -u interpretation_features.py SVM_models/SVM_model_ngrams_tf_2024-05-23 svm ngrams -ftp ngram_features/tf/ngram_frequencies_spmatrix_tf_dev >> out.txt
# # python3 -u interpretation_features.py SVM_models/SVM_model_ngrams_tf_nofeat_2024-05-23 svm nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev >> out.txt
# # python3 -u interpretation_features.py DT_models/DT_model_ngrams_tf_2024-05-22 dt ngrams -ftp ngram_features/tf/ngram_frequencies_spmatrix_tf_dev >> out.txt
# # python3 -u interpretation_features.py DT_models/DT_model_ngrams_tf_nofeat_2024-05-22 dt nofeat -ftp ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_dev >> out.txt
# rm -f out.txt
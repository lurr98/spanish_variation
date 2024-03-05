import os, re, zipfile

path_to_folder = '/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS'
which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
out_folder = '/projekte/semrel/WORK-AREA/Users/laura/'

for zip_file in os.listdir(path_to_folder):
    # select only the zip files for the specified classes i.e. countries
    if zip_file.endswith('.zip') and re.split('_|-', zip_file)[1] in which_country:
        # use zipfile to extract the files from the desired zip files, every zip file contains the corpus for one class
        with zipfile.ZipFile('{}/{}'.format(path_to_folder, zip_file), 'r') as zipref:
            for file in zipref.namelist()[:1]:
                with zipref.open(file, 'r') as f:
                    file_content = f.read()

                with zipref.open(out_folder + file, 'w') as f:
                    file_content = f.write()
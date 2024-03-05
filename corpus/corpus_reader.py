# TODO: define class of CorpusReader which can
#
#       - read the data for a specific class of the corpus
#           - take into account that the separate classes are split into multiple files as well 
#       - read the data for all desired classes at once
#       - assign labels
#       - split the data into paragraphs / chunks / sentences (?)
#       - preprocess the data ?

import os, re, zipfile


class CorpusReader:
# define the corpus reader class that will be used by the models to access data

    def __init__(self, path_to_folder, which_country, chunking_type):

        def append_id_data(id_data, appropriate_line):
            # helper function to make code look better (and hopefully more efficient)
            for i, inner_list in enumerate(id_data):
                inner_list.append(appropriate_line[i])
            return id_data

        # define a dictionary that will later contain the class tag as key and the corresponding data as value
        all_data, all_ids = {}, {}

        for zip_file in os.listdir(path_to_folder):
            # select only the zip files for the specified classes i.e. countries
            if zip_file.endswith('.zip') and re.split('_|-', zip_file)[1] in which_country:
                class_data = {}
                # use zipfile to extract the files from the desired zip files, every zip file contains the corpus for one class
                with zipfile.ZipFile('{}/{}'.format(path_to_folder, zip_file), 'r') as zipref:
                    for file in zipref.namelist():
                        with zipref.open(file, 'r') as f:
                            lines = f.readlines()

                        # decode each line bc they are currently bytes
                        lines = [line.decode('latin-1') for line in lines]
                        # if we want to chunk the text using the predefined paragraphs
                        if chunking_type == 'pars':

                            for line in lines:
                                # if the line defines the start of a paragraph using its id
                                if re.search('@@\d+', line.split('\t')[2]):
                                    # define exception in case this is the first paragraph
                                    try:
                                        # add data from previous paragraph to dict, add name of file for better mapping
                                        class_data['{}_{}'.format(file.split('.')[0], id_num)] = id_data
                                    except NameError:
                                        pass
                                    # assign the id to a variable
                                    id_num = line.split('\t')[0]
                                    # three inner lists for token, lemma and pos respectively
                                    id_data = [[], [], []]
                                else:
                                    # for every line that is not the paragraph's id, append the content to the appropriate lists
                                    id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], line.split('\t')[4]])
                                    # id_data[0].append(line.split('\t')[2])
                                    # id_data[1].append(line.split('\t')[3])
                                    # id_data[2].append(line.split('\t')[4])

                        # if we want to chunk the text by sentences
                        # ! this is a very simple approach, better to use a sentence splitter?
                        if chunking_type == 'sents':

                            for i, line in enumerate(lines):
                                # if the line defines the start of a paragraph using its id
                                if re.search('@@\d+', line.split('\t')[0]):
                                    # define exception in case this is the first paragraph
                                    try:
                                        # in case there was no delimiter before the start of the new paragraph add the sentence's data to the dict 
                                        if not sentence_id in all_ids: 
                                            class_data['{}_{}'.format(file.split('.')[0], sentence_id)] = id_data
                                            sentence_id += 1
                                    except NameError:
                                        # generate a sentence id
                                        sentence_id = 0
                                        # three inner lists for token, lemma and pos respectively
                                        id_data = [[], [], []]
                                # if the line contains a period as a punctuation mark
                                elif line.split('\t')[2] == '.' and line.split('\t')[3] == '$.' and line.split('\t')[4] == 'y':
                                    # check whether the next token starts with an uppercase letter
                                    if lines[i+1].split('\t')[2][0].isupper():
                                        # add the delimiter to the sentence's data
                                        id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], line.split('\t')[4]])
                                        # add all the data to the sentence id in the dict, add file name for better mapping
                                        class_data['{}_{}'.format(file.split('.')[0], sentence_id)] = id_data
                                        # add one to sentence id to start a new sentence
                                        sentence_id += 1
                                        # empty data from previous sentence
                                        id_data = [[], [], []] 
                                    else:
                                        # add the delimiter to the sentence's data and proceed
                                        id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], line.split('\t')[4]]) 
                                else:
                                    # for every line that is not a delimiter, append the content to the appropriate lists
                                    id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], line.split('\t')[4]])

                all_data[re.split('_|-', zip_file)[1]] = class_data
                all_ids[re.split('_|-', zip_file)[1]]  = list(class_data.keys())          

        self.data = all_data
        self.ids = all_ids



if __name__ == "__main__":
    cr = CorpusReader('../../corpus', ['PA'], 'pars')

    print(type(cr.data))
    print(list(cr.data['PA'].items())[-2:])

    print(type(cr.ids))
    print(cr.ids['PA'][:10])
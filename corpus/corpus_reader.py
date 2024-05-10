# TODO: define class of CorpusReader which can
#
#       - read the data for a specific class of the corpus
#           - take into account that the separate classes are split into multiple files as well 
#       - read the data for all desired classes at once
#       - assign labels
#       - split the data into paragraphs / chunks / sentences (?)
#       - preprocess the data ?

import os, re, zipfile, string, json, random


# global variable so that this doesn't have to be passed around all the time
with open('../corpus/POS_related/inverted_POS_tags.json', 'r') as jsn:
    POS_mapping = json.load(jsn)


class CorpusReader:
# define the corpus reader class that will be used by the models to access data

    def __init__(self, path_to_folder: str, which_country: list, chunking_type: str, filter_punct: bool=False, lower: bool=False, split_data: list=[0.8, 0.9], sub_sample: bool=True):

        saved_args = locals()
        print('CorpusReader was initialised with the following arguments: {}'.format(saved_args))

        def append_id_data(id_data: list, appropriate_line: list, lower: str) -> list:
            # helper function to make code look better (and hopefully more efficient)
            for i, inner_list in enumerate(id_data):
                if lower:
                    inner_list.append(appropriate_line[i].lower())
                else:
                    inner_list.append(appropriate_line[i])

            return id_data
        
        def substitute_null_char(pos: str) -> str:

            if pos.strip() == '\x00':
                return 'x00'
            else:
                return pos.strip()

        # define a dictionary that will later contain the class tag as key and the corresponding data as value
        all_data, all_raw_data, all_ids, all_tokens = {}, {}, {}, {}

        if split_data:
            # if we want to get the data splits, namely the three lists containing the corresponding ids
            train, test, dev = {}, {}, {}

        for zip_file in os.listdir(path_to_folder):
            # select only the zip files for the specified classes i.e. countries
            if zip_file.endswith('.zip') and re.split('_|-', zip_file)[1] in which_country:
                print('Unzipping corpus for dialect {}\n--------------------------------------------------------\n'.format(re.split('_|-', zip_file)[1]))
                class_data, raw_class_data, class_tokens = {}, {}, 0
                # use zipfile to extract the files from the desired zip files, every zip file contains the corpus for one class
                with zipfile.ZipFile('{}/{}'.format(path_to_folder, zip_file), 'r') as zipref:
                    for i, file in enumerate(zipref.namelist()):
                        if i != 0 and i % 5 == 0:
                            print('Working on file {} out of {}'.format(i, len(zipref.namelist())))
                        with zipref.open(file, 'r') as f:
                            lines = f.readlines()

                        # decode each line bc they are currently bytes
                        try:
                            lines = [line.decode('utf-8') for line in lines]
                        except UnicodeDecodeError:
                            lines = [line.decode('latin-1').encode().decode('utf-8') for line in lines]

                        # if we want to chunk the text using the predefined paragraphs
                        if chunking_type == 'pars':

                            for line in lines:
                                # if the line defines the start of a paragraph using its id
                                if re.search('@@\\d+', line.split('\t')[2]):
                                    # define exception in case this is the first paragraph
                                    try:
                                        # add data from previous paragraph to dict, add name of file for better mapping
                                        class_data['{}_{}'.format(file.split('.')[0], id_num)] = id_data
                                        # add number of tokens
                                        class_tokens += len(id_data[0])
                                        # reconstruct original format but wihtout the id etc. but with the addition of the POS mapping
                                        line_data = ['{}\t{}\t{}\t{}'.format(token, id_data[1][j], id_data[2][j], id_data[3][j]) for j, token in enumerate(id_data[0])]
                                        if lower:
                                            raw_class_data['{}_{}'.format(file.split('.')[0], id_num)] = '\n'.join(line_data).lower()
                                        else:
                                            raw_class_data['{}_{}'.format(file.split('.')[0], id_num)] = '\n'.join(line_data)
                                    except NameError:
                                        pass
                                    # assign the id to a variable
                                    id_num = line.split('\t')[0]
                                    # three inner lists for token, lemma and pos respectively
                                    id_data = [[], [], [], []]
                                else:
                                    if filter_punct:
                                        # get rid of punctuation
                                        if not line.split('\t')[2] in set(string.punctuation):
                                            # make sure the POS tag is not the null character
                                            pos_tag = substitute_null_char(line.split('\t')[4].strip())
                                            # for every line that is not the paragraph's id, append the content to the appropriate lists
                                            id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], pos_tag, POS_mapping[pos_tag.lower()]], lower)
                                    else:
                                        # make sure the POS tag is not the null character
                                        pos_tag = substitute_null_char(line.split('\t')[4].strip())
                                        # for every line that is not the paragraph's id, append the content to the appropriate lists
                                        id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], pos_tag, POS_mapping[pos_tag.lower()]], lower)

                        # TODO: check whether I really need this and if so, update (filter_punct etc.)
                        # TODO: if needed add raw_data as well!
                                        
                        # # if we want to chunk the text by sentences
                        # # ! this is a very simple approach, better to use a sentence splitter?
                        # if chunking_type == 'sents':

                        #     for i, line in enumerate(lines):
                        #         # if the line defines the start of a paragraph using its id
                        #         if re.search('@@\\d+', line.split('\t')[0]):
                        #             # define exception in case this is the first paragraph
                        #             try:
                        #                 # in case there was no delimiter before the start of the new paragraph add the sentence's data to the dict 
                        #                 if not sentence_id in all_ids: 
                        #                     class_data['{}_{}'.format(file.split('.')[0], sentence_id)] = id_data
                        #                     sentence_id += 1
                        #             except NameError:
                        #                 # generate a sentence id
                        #                 sentence_id = 0
                        #                 # three inner lists for token, lemma and pos respectively
                        #                 id_data = [[], [], []]
                        #         # if the line contains a period as a punctuation mark
                        #         elif line.split('\t')[2] == '.' and line.split('\t')[3] == '$.' and line.split('\t')[4] == 'y':
                        #             # check whether the next token starts with an uppercase letter
                        #             if lines[i+1].split('\t')[2][0].isupper():
                        #                 # add the delimiter to the sentence's data
                        #                 id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], line.split('\t')[4], POS_mapping[line.split('\t')[4]]])
                        #                 # add all the data to the sentence id in the dict, add file name for better mapping
                        #                 class_data['{}_{}'.format(file.split('.')[0], sentence_id)] = id_data
                        #                 # add one to sentence id to start a new sentence
                        #                 sentence_id += 1
                        #                 # empty data from previous sentence
                        #                 id_data = [[], [], []] 
                        #             else:
                        #                 # add the delimiter to the sentence's data and proceed
                        #                 id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], line.split('\t')[4], POS_mapping[line.split('\t')[4]]]) 
                        #         else:
                        #             # for every line that is not a delimiter, append the content to the appropriate lists
                        #             id_data = append_id_data(id_data, [line.split('\t')[2], line.split('\t')[3], line.split('\t')[4], POS_mapping[line.split('\t')[4]]])


                all_data[re.split('_|-', zip_file)[1]] = class_data
                all_raw_data[re.split('_|-', zip_file)[1]] = raw_class_data
                all_ids[re.split('_|-', zip_file)[1]] = list(class_data.keys())
                all_tokens[re.split('_|-', zip_file)[1]] = class_tokens 

                print('\n--------------------------------------------------------\n')

        if split_data:
            min_samples_prox = min([len(idx_list) for idx_list in list(all_ids.values())])

            for label, class_ids in all_ids.items():
                print(label)
                if sub_sample:
                    # if sub-sampling is enabled, make the minimum number of samples the number of documents of the smallest subcorpus 
                    min_samples = min_samples_prox
                else:
                    min_samples = len(class_ids)

                # shuffle the ids to maximise randomness
                random.shuffle(class_ids)

                # this either shortens the index list or leaves it alone (in case of no sub-sampling)
                class_ids_sampled = class_ids[:min_samples]

                split_train = int(split_data[0]*len(class_ids_sampled))
                split_test_dev = int(split_data[1]*len(class_ids_sampled))

                # fill the train, dev, test dictionaries with the appropriate number of ids for each set
                # TODO: do we need some algorithm to fix imbalance 
                train[label] = class_ids_sampled[:split_train]
                test[label] = class_ids_sampled[split_train:split_test_dev]
                dev[label] = class_ids_sampled[split_test_dev:]

            self.train = train
            self.test = test
            self.dev = dev

        self.data = all_data
        self.raw = all_raw_data
        self.ids = all_ids
        self.number_of_tokens = all_tokens



if __name__ == "__main__":
    cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE'], 'pars')

    print(type(cr.data))
    print(list(cr.data.keys()))
    print(cr.ids)
    # print(cr.ids['AR'][:10])
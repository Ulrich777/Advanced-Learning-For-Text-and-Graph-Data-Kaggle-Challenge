from helper import load_data, tensorize_graphs, load_docs

path_to_data = "C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/for_kaggle_final/work/"

graphs, embds, ADJ_OPT, EMB_OPT, ADJ_TEST, EMB_TEST, targets = load_data(nmax=29, RESET=True, 
                                                                         path_to_data=path_to_data)

del ADJ_OPT, EMB_OPT, ADJ_TEST, EMB_TEST
ADJ, EMB = tensorize_graphs(graphs, embds, nmax=29)

docs = load_docs(BUILD_DOCS = True, pad_vec_idx = 1685894, num_walks = 5, 
              walk_length = 10, max_doc_size = 70, path_to_data=path_to_data)


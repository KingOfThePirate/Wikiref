import os,sys,pickle

def save_pickle(object_to_save,filename):
	fileObject = open(filename,'wb')
	pickle.dump(object_to_save,fileObject)
	fileObject.close()

def load_pickle(filename):
	fileObject = open(filename,'rb')
	b = pickle.load(fileObject)
	fileObject.close()
	return b

out_link_dict_sep = load_pickle("out_link_dict_sep")

out_link_dict_combine = dict()

for A in out_link_dict_sep:
	out_link_dict_combine[A] = dict()
	for B in out_link_dict_sep[A]:
		sent_comb_a = ""
		sent_comb_b = ""
		for intersect in out_link_dict_sep[A][B] :
			A_sent,B_sent= out_link_dict_sep[A][B][intersect].split("##@@##")
			sent_comb_a = sent_comb_a + A_sent
			sent_comb_b = sent_comb_b + B_sent
		out_link_dict_combine[A][B] = sent_comb_a + "##@@##" + sent_comb_b

save_pickle(out_link_dict_combine,"out_link_dict_combine")

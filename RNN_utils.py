from __future__ import print_function
import numpy as np
import pickle
# method for generating text
def generate_text(model, length, vocab_size, ix_to_char,char_to_ix,STR_N):
	# starting with random character
	b=STR_N
	
	a= [ b[i] for i in range(len(b)) ]
	a.append(' ')
	
	ix=[char_to_ix[i] for i in a]
	y_char = a
	X = np.zeros((1, length, vocab_size))
	print(STR_N,end=" ")
	for i in range(0,len(b)+1):
                X[0, i, :][ix[i]] = 1
                
                
	for i in range(len(b)+2,length):
		# appending the last predicted character to sequence
		X[0, i, :][ix[-1]] = 1
		print(ix_to_char[ix[-1]], end="")
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
		
	return ('').join(y_char)

# method for preparing the training data
def load_data(data_dir, seq_length):
	data = open(data_dir, 'r').read()
	chars = list(set(data))
	VOCAB_SIZE = len(chars)
	

	print('Data length: {} characters'.format(len(data)))
	print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

	with open('misc_data.pickle','wb') as fp:
                pickle.dump([ix_to_char,char_to_ix,VOCAB_SIZE],fp)

	X = np.zeros((int(len(data)/seq_length), seq_length, VOCAB_SIZE))
	y = np.zeros((int(len(data)/seq_length), seq_length, VOCAB_SIZE))
	for i in range(0, int(len(data)/seq_length)):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
			X[i] = input_sequence

		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
	return X, y, VOCAB_SIZE, ix_to_char,char_to_ix

load_data('H:\\G_drive\\AI\\text-generator-master\\data\\data.txt',50)

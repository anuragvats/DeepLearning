from __future__ import print_function
import numpy as np

# method for generating text
def generate_text(model, length, vocab_size, char_to_ix,ix_to_char):
	# starting with random character
	ix = ['Endnu']
	#y_char = [ix_to_char[ix[-1]]]
	X = np.zeros((1,length, vocab_size))
	for i in range(length):
                for j in [ix[i][k]for k in range(len(ix[-1]))]:
                    
                    X[0][i][char_to_ix[j]]=1

                
                
                x = model.predict(X[:, :i+1, :])[0]
                
                a=''
                for l in range(vocab_size):
                    if x[i][l]>.008 :
                        x[i][l]=1
                        a=a+ix_to_char[l]
                        
                    else :
                        x[i][l]=0

                X[0][i]=x[i]
                ix.append(a)
                print(a,end=" ")
		
		
		
	return ('')

# method for preparing the training data
def load_data(pos):
	
	data=open(pos,'r').read()
	chars = list(set(data))
	VOCAB_SIZE = len(chars)

	

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}
	
	no_string=0
	
	with open(pos,'r') as f:
            contents = f.readlines()
            for line in contents:
                    no_string=no_string+1

        
	X = np.zeros((1,no_string,VOCAB_SIZE))
	y = np.zeros((1,no_string,VOCAB_SIZE))
	print(no_string)

	i=0

	with open(pos,'r') as f:
            contents = f.readlines()
            
            for line in contents:
                
                for c in line.split():
                    #print(c)
                    for j in [c[l] for l in range(len(c))]:
                        
                        X[0][i][char_to_ix[j]]=1
                i=i+1
                        


	i=0
	with open(pos,'r') as f:
            contents = f.readlines()
            
            for line in contents[1:]:
                for c in line.split():
                    for j in [c[l] for l in range(len(c))]:
                        #print(j)
                        y[0][i][char_to_ix[j]]=1
                i=i+1
                        

        
        
	
	
	return X, y, VOCAB_SIZE, char_to_ix,ix_to_char

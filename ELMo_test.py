#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In TF: https://github.com/tks10/ELMo

# https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L27
from allennlp.modules.elmo import Elmo, batch_to_ids


# In[2]:


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, num_output_representations=4, dropout=0.1)


# In[5]:


# use batch_to_ids to convert sentences to character ids
# sentences = [['First', 'sentence', '.'], ['Another', 'sentence','is','this','.'],['A', 'good','sentence','.']]
sentences = [["I", "ate", "an", "apple", "for", "breakfast"],["I", "ate", "an", "orange", "for", "dinner"]] 
#Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters (len(batch), max sentence length, max word length).
character_ids = batch_to_ids(sentences)

print (character_ids,character_ids.size())
embeddings = elmo(character_ids)


# In[7]:


emb=embeddings['elmo_representations']
print (type(emb))
print (len(emb)) 
# List[torch.Tensor]


# In[8]:


emb[1].size()  #(batch_size, timesteps, embedding_dim)


# In[11]:


# the lower layer 0;  for the word  'ate'
s1 = emb[0][0][1]
s2 = emb[0][1][1]
print (s1, s1.size())


# In[10]:


# calculate the word 'sentence' in both cases, the lower layer(indexed by 0)
import scipy
scipy.spatial.distance.cosine(s1.detach().numpy(),s2.detach().numpy())


# In[ ]:


s1 = emb[3][0][1]
s2 = emb[3][1][1]
scipy.spatial.distance.cosine(s1.detach().numpy(),s2.detach().numpy())


# In[ ]:


s1


# In[12]:


# if you want a numpy...
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np


# In[13]:


elmo = ElmoEmbedder() 
vector1 = elmo.embed_sentence(sentences[0])
vector2 = elmo.embed_sentence(sentences[1])


# In[14]:


np.shape(vector1)


# In[15]:


scipy.spatial.distance.cosine(vector1[0][-1],vector2[0][-1])


# In[ ]:


# 1. Use a single layer
#sents = elmo_embedding['elmo_representations'][-1]


#2.  Concate all ELMo layers
#sent_list = [vect for vect in elmo_embedding['elmo_representations']]
#sents = torch.cat(sent_list,2).view(batch_size,-1,self.embedding_dim, self.elmo_level)


# (maybe) A weighted sum of all the layers
#vars = torch.Tensor(self.elmo_level,1).cuda()
#sents = torch.matmul(sents,vars).view(batch_size,-1,self.embedding_dim)

# 3. Concate with your embedding
#embedded = nn.Embedding(vocab_size,embedding_dim) # your embedding...
#sents = torch.cat((sents[:, :time_step_emb], embedded), 2) # concate with ELMo 


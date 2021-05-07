import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

# Problem 1 ######################## see : https://github.com/pytorch/benchmark/blob/master/rnns/benchmarks/lstm_variants/lstm.py
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()

    # TODO ========================
    # Initialization of the parameters of the recurrent and fc layers. 
    # Your implementation should support any number of stacked hidden layers 
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules 
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
    # provided clones function (as opposed to a regular python list), in order 
    # for Pytorch to recognize these parameters as belonging to this nn.Module 
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.emb_size = emb_size###
    
    self.input = nn.Embedding(vocab_size, emb_size)
    self.dropout_input = nn.Dropout(p=dp_keep_prob)
    
    self.activation = nn.Tanh()
    
    self.recurrent = nn.ModuleList()
    for i in range(num_layers):        
        input_size = emb_size if i == 0 else hidden_size
        module_list = nn.ModuleList()
        module_list.add_module("i2h", nn.Linear(input_size, hidden_size))
        module_list.add_module("Dropout", nn.Dropout(p=dp_keep_prob))
        module_list.add_module("h2h", nn.Linear(hidden_size, hidden_size))
        self.recurrent.add_module("layer_" + str(i), module_list)
    
    self.output = nn.Linear(hidden_size,vocab_size)
    self.init_weights_uniform()
    
  def init_weights_uniform(self):
    # TODO ========================
    # Initialize all the weights and bias uniformly as default
    u = math.sqrt(1/self.hidden_size)
    for name, p in self.named_parameters():
        if 'weight' in name or 'bias' in name:
            torch.nn.init.uniform_(p, -u, u)
    
    # Initialize all the weights uniformly in the range [-0.1, 0.1]
    # and all the biases to 0 (in place) for input and output layer
    for name, p in self.input.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(p,-0.1,0.1)
        elif 'bias' in name:
            nn.init.constant_(p, 0)
    
    for name, p in self.input.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(p,-0.1,0.1)
        elif 'bias' in name:
            nn.init.constant_(p, 0)
 

  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    h = torch.zeros([self.num_layers,self.batch_size,self.hidden_size])
    return h # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # TODO ========================
    # Compute the forward pass, using a nested python for loops.
    # The outer for loop should iterate over timesteps, and the 
    # inner for loop should iterate over hidden layers of the stack. 
    # 
    # Within these for loops, use the parameter tensors and/or nn.modules you 
    # created in __init__ to compute the recurrent updates according to the 
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
    
    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    
    embedded_inputs = self.input(inputs)
    
    
    logits = torch.tensor([])
    for t in range(self.seq_len):
        # prediction calculation :
        layer_inputs = self.dropout_input(embedded_inputs[t])
        for i in range(self.num_layers):
            m = eval('self.recurrent.layer_' + str(i))
                    #hidden[i] = torch.tanh(input_i2h + torch.mm(hidden[i],m.h2h.weight.data) + m.h2h.bias.data)
            hidden[i] = self.activation(m.h2h(hidden[i].clone()) + m.i2h(layer_inputs))
            layer_inputs = m.Dropout(hidden[i].clone())
                    #input_i2h = torch.mm(input_i2h.data,torch.transpose(m.h2out.weight.data, 0, 1)) + m.h2out.bias.data
        out = self.output(layer_inputs)
        logits = torch.cat((logits, (out)), 0)
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden


####################################################################################


  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    # Compute the forward pass, as in the self.forward method (above).
    # You'll probably want to copy substantial portions of that code here.
    # 
    # We "seed" the generation by providing the first inputs.
    # Subsequent inputs are generated by sampling from the output distribution, 
    # as described in the tex (Problem 5.3)
    # Unlike for self.forward, you WILL need to apply the softmax activation 
    # function here in order to compute the parameters of the categorical 
    # distributions to be sampled from at each time-step.

    """
    Arguments:
        - input: A mini-batch of input tokens (NOT sequences!)
                        shape: (batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
        - generated_seq_len: The length of the sequence to generate.
                       Note that this can be different than the length used 
                       for training (self.seq_len)
    Returns:
        - Sampled sequences of tokens
                    shape: (generated_seq_len, batch_size)
    """
    samples = torch.zeros([generated_seq_len, self.batch_size])
    samples = samples.to(torch.device("cuda"))
    # ex_hid = hidden
    samples[0,:] = input
    
    for module in range(generated_seq_len):
        #current_hid = []
        # hid_out = self.init_hidden().to(torch.device('cuda'))
        in_to_cell = self.encoder(input)
        in_to_cell = self.drop(in_to_cell)
        for layer in range(self.num_layers):
            hid_temp = self.rec_layers[layer](hidden[layer,:,:])
            in_to_cell = (self.regular_layers[layer](in_to_cell) + hid_temp)
            in_to_cell = torch.tanh(in_to_cell)
            hidden[layer,:,:] = in_to_cell
            self.drop(in_to_cell)
        output = self.decoder(in_to_cell)
        output = F.softmax(output, dim=0)
        output = torch.multinomial(output, 1)
        samples[module,:] = output.squeeze()
    return samples

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

# Problem 2
   
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py

  """
  Follow the same instructions as for RNN (above), but use the equations for 
  GRU, not Vanilla RNN.
 
  For each element in the input sequence, each layer computes the following
    function:
    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            \hat{h}_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * \hat{h}_t + z_t * h_{(t-1)}
        \end{array}
  """

      ##############################################################################################
'''
#  @article{cho2014learning,
#  title={Learning phrase representations using RNN encoder-decoder for statistical machine translation},
#  author={Cho, Kyunghyun and Van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
#  journal={arXiv preprint arXiv:1406.1078},
#  year={2014}
#}
  
'''
    
class GRU_Cell(nn.Module):
    
    def __init__(self, input_size, hidden_size, dp_keep_prob):
        super(GRU_Cell, self).__init__()
        
        self.dropout = torch.nn.Dropout(1 - dp_keep_prob)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()  
        self.hidden_size = hidden_size
        
        self.w_r = torch.nn.Linear(input_size,  hidden_size)
        self.w_z = torch.nn.Linear(input_size,  hidden_size)
        self.w_h= torch.nn.Linear(input_size,  hidden_size)
    
        self.u_r = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_z = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_h = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.b_r = torch.nn.Linear(1,hidden_size)
        self.b_z = torch.nn.Linear(1,hidden_size)
        self.b_h = torch.nn.Linear(1,hidden_size)
                  
    def forward(self, in_to_cell, hid_of_cell):
        
        r_t = self.sigm(self.w_r(in_to_cell) + self.u_r(hid_of_cell))
        z_t = self.sigm(self.w_z(in_to_cell) + self.u_z(hid_of_cell))
        h_hat = self.tanh(self.w_h(in_to_cell) + self.u_h(r_t*hid_of_cell))
        h_t = ((1-z_t)*hid_of_cell) + (z_t*h_hat) 
        y = self.dropout(h_t)
            
        return h_t, y
      
      
#####################################################################################################

class GRU(nn.Module): # Implement a stacked GRU RNN
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(GRU, self).__init__()
        # TODO ========================
 
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob  
        self.tanh = torch.nn.Tanh()
        self.sigm = torch.nn.Sigmoid()
        ##       self.bias = bias
        self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_size)
        self.dropout = torch.nn.Dropout(p=(1 - dp_keep_prob))
        self.w_y = torch.nn.Linear(hidden_size, vocab_size)
        self.layers = torch.nn.ModuleList()
        
        for module in range(1, self.num_layers + 1):
            [GRU_Cell(hidden_size, emb_size, dp_keep_prob)].append(GRU_Cell(hidden_size, hidden_size, dp_keep_prob))
        
        self.hidden_stack = nn.ModuleList([GRU_Cell(hidden_size, emb_size, dp_keep_prob)])
        self.output = torch.nn.Linear(hidden_size, self.vocab_size)
        self.init_weights_uniform()
        
    def init_weights_uniform(self):
        # TODO ========================
        u = math.sqrt(1/self.hidden_size)
        for module in [self.w_r, self.u_r, self.b_r, self.w_z, self.u_z, self.b_z, self.w_h, self.u_h, self.b_h]:
            torch.nn.init.uniform_(module, -u, u)
    
    def init_hidden(self):
        # TODO ==============================================================
        return torch.zeros([self.num_layers, self.batch_size, self.hidden_size])
        # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        
    def forward(self, inputs, hidden):
        # See details in https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
            # TODO ========================================================
        '''
        input.shape()   =&gt; (batch_size, input_size)
        gru_out.shape() =&gt; (seq_len, batch_size, hidden_size)
        outputs.shape() =&gt; (seq_len, batch_size, output_size)
        '''
        # TODO ========================
        logits = []
        hid_of_cell_list = []
        embs = self.embedding(inputs)
        # shape: (self.seq_len, self.batch_size, self.emb_size)
  
        for module in embs:
            layer_to_out = self.dropout(module)
            for next_layer, idx  in enumerate(self.module_list):
                ex_hid_layer = hidden[idx]
                hid_layer = hidden(layer_to_out, ex_hid_layer)
                layer_to_out = self.dropout(hid_layer)
                hid_of_cell_list.append(hid_layer)
            
            hid_of_cell = torch.stack(hid_of_cell_list)
            logits.append(self.output(layer_to_out))
            
        logits = torch.stack(logits)
        return logits, hid_of_cell
        
    
#    r_t = self.sigm(self.w_r(in_to_cell) + self.u_r(hid_of_cell))
#    z_t = self.sigm(self.w_z(in_to_cell) + self.u_z(hid_of_cell))
#    h_hat = self.tanh(self.w_h(in_to_cell) + self.u_h(r*hid_of_cell))
#    h_t = ((1-z_t)*hid_of_cell) + (z_t*h_hat) 
#    y = self.dropout(h_t)

    
#    return h_t, y

    def generate(self, input, hid_to_cell, generated_seq_len):
    # TODO ========================
    
        for module in range(generated_seq_len):
            current_hid = []
            in_to_cell = self.embs(input)
        
            for i in range(self.num_layers):
                gru_drop, gru_cell = self.module_list[i]
                out_to_drop = gru_drop(in_to_cell)
                out_to_cell = gru_cell(out_to_drop, hid_to_cell[i])
                current_hid.append(out_to_cell)
                in_to_cell = out_to_cell
            
            out_fc_block = self.fc(in_to_cell)
            hid_to_cell = current_hid
            return torch.multinomiql(torch.nn.functional.softmax(out_fc_block), 1)

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


#----------------------------------------------------------------------------------
# Based on http://nlp.seas.harvard.edu/2018/04/03/attention.html -------"Attention is You You Need"--------
# Based on https://github.com/dreamgonfly/Transformer-pytorch/blob/master/models.py
'''
@inproceedings{rush2018annotated,
  title={The Annotated Transformer},
  author={Rush, Alexander},
  booktitle={Proceedings of Workshop for NLP Open Source Software (NLP-OSS)},
  pages={52--60},
  year={2018}
}
'''
#
# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0, \
           "Please notice that there is a problem in the devision n_heads (given:{}) divides n_units (given:{}).".format(
               n_heads, n_units)
        self.n_units = n_units
        self.n_heads = n_heads
        
        """
         TODO: 
         create and/or 
         initialize any necessary parameters or layers
         Note: the only Pytorch modules you are allowed to use are 
         nn.Linear 
         and 
         nn.Dropout
        """        
        # build layers
        # an affine operation: y = Wx + b        
        
        #self.linears = clones(nn.Linear(n_units, n_units), 4)
        
        self.w_q = nn.Linear(self.n_units, self.n_units)
        # self.w_q = clones(self.w_q, n_heads)
        # self.w_1 = nlp.nmt.onmt.modules.BottleLinear(size, hidden_size)
        
        self.w_k = nn.Linear(self.n_units, self.n_units)
        # self.w_k = clones(self.w_k, n_heads)
        
        self.w_v = nn.Linear(self.n_units, self.n_units)
        # self.w_v = clones(self.w_v, n_heads)
        
        self.w_o = nn.Linear(self.n_units, self.n_units)
        self.dropout= nn.Dropout(dropout)
        
        # initialize any necessary parameters or layers
        
        u = math.sqrt(1/n_units)
        for module in self.w_q, self.w_k, self.w_v, self.w_o :
            torch.nn.init.uniform_(module.weight, -u, u)   
            torch.nn.init.uniform_(module.bias, -u, u)
            return
        
        
    # define function "attntion"
    # Compute 'Scaled Dot Product Attention
    def attn(self, key, query, value, mask=None, dropout=None):
            x = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
            
            if mask is not None:
                mask = mask.float().unsqueeze(1).expand(-1, x.shape[1], -1, -1)
#               x = x.masked_fill(mask == 0, -1e9)
                x = (x * mask) - ((1-mask)*(1e9))
                p_attn = F.softmax(x, dim=-1)
            if dropout is not None:
                p_attn = dropout(p_attn)
                output = torch.matmul(p_attn, value)
                return output

    ##########################
    
    
        # method forward to compute the network output
    def forward(self, query, key, value, mask=None):
        """
         TODO: 
         implement the masked multi-head attention.
         query, 
         key, and 
         value 
         all have size: (batch_size, seq_len, self.n_units)
         mask has size: (batch_size, seq_len, seq_len)
         As described in the .tex, 
         apply input masking to the softmax 
         generating the "attention values" (i.e. A_i in the .tex)
         Also apply dropout to the attention values.
        """
        
#        if mask is not None:
#              # Same mask applied to all h heads.
#              mask = mask.unsqueeze(1)
              
        ########################################################################
        batch_size = query.size(0)
        seq_len = query.size(1)
        query = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        key   = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        value = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        
        # transpose to get dimensions batch_size * n_heads * sl * n_units
     
        scores = self.attn(key, query, value, mask=mask, dropout=self.dropout)
         
#        "Concat" using a view and apply a final linear
        scores = scores.transpose(1,2).contiguous().view(batch_size, seq_len, self.n_units)
        output = self.w_o(scores)
#        
        return output

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


#my_cool = RNN(12,4,10,3,1000,2,0.5)

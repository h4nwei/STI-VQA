import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class dGRUCell(nn.Module):
    """
    use more parameters to fit the difference GRU model
    """

    def __init__(self, input_size=128, hidden_size=32, bias=True):
        super(dGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.diff_0 = nn.Linear(input_size, input_size, bias=bias) #128
        self.diff = nn.Linear(input_size, 3 * hidden_size, bias=bias) #128
        self.hx_0 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hx = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.whx_0 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.whx = nn.Linear(hidden_size, input_size, bias=bias)
        self.wx1_0 = nn.Linear(input_size, input_size, bias=bias)
        self.wx1 = nn.Linear(input_size, input_size, bias=bias)

        self.x1_0 = nn.Linear(input_size, input_size, bias=bias)
        self.x1 = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.dy_0 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dy = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        # self.bn_input = nn.BatchNorm1d(num_features=input_size)
        # self.bn_hidden = nn.BatchNorm1d(num_features=hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x1, x0, hidden):
        
        dx, hx = hidden
        x1 = x1.view(-1, x1.size(1))
        x0 = x0.view(-1, x0.size(1))

        # memory the difference between current frame and last frame
        # diff = x1 - x0
        # w_diff = torch.softmax(self.hxx1(self.hxx1_0(torch.cat([hx, x1], dim=1)))) * diff
        w_diff = torch.softmax(self.whx(self.whx_0(hx)) + self.wx1(self.wx1_0(x1)), dim=1) * x0
        gates_d = self.hx(self.hx_0(hx)) + self.diff(self.diff_0(w_diff))

        resetgate_d, cellgate_d, outgate_d = gates_d.chunk(3, 1)
        
        resetgate_d = torch.sigmoid(resetgate_d)
        cellgate_d = torch.tanh(cellgate_d)
        outgate_d = torch.sigmoid(outgate_d)
        
        dy = torch.mul(dx, (1 - resetgate_d)) +  torch.mul(resetgate_d, cellgate_d)        
        hy = torch.mul(outgate_d, dy)
        
        # memory the distortion of current frame
        gates_h = self.x1(self.x1_0(x1)) + self.dy(self.dy_0(dy))
        resetgate_h, cellgate_h = gates_h.chunk(2, 1)
        
        resetgate_h = torch.sigmoid(resetgate_h)
        cellgate_h = torch.tanh(cellgate_h)
        # outgate_h = torch.sigmoid(outgate_h)
        
        cz= torch.mul(hy, (1 - resetgate_h)) +  torch.mul(resetgate_h, cellgate_h)        
        # dz = torch.mul(outgate_x, cz)
        
        return (dy, cz)

'''
STEP 3: CREATE MODEL CLASS
'''
 
class dGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1, bias=True):
        super(dGRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
         
        # Number of hidden layers
        self.layer_dim = layer_dim
               
        self.dGRU = dGRUCell(input_dim, hidden_dim, layer_dim)  
        self.fc_h1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_h2 = nn.Linear(hidden_dim//2, output_dim)
        self.fc_d1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_d2 = nn.Linear(hidden_dim//2, output_dim)
     
    
    
    def forward(self, x1, x0):
        '''
        Input:
        x1: the features of the current frames
        x0: the feaures of the last frames
        '''
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            d0 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim).cuda())
            h0 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim).cuda()) # Initialize cell state
        else:
            d0 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim))
            h0 = Variable(torch.zeros(self.layer_dim, x1.size(0), self.hidden_dim)) # Initialize cell state
       
        outs_d = []
        outs_h = []

        dn = d0[0,:,:] # short term memory
        hn = h0[0,:,:] # long term memory


        for seq in range(x1.size(1)):
            dn, hn = self.dGRU(x1[:,seq,:], x0[:,seq,:], (dn,hn)) 
            # print('x1[:,seq,:]', x1[:,seq,:])
            # print('x0[:,seq,:]', x0[:,seq,:])
            outs_d.append(dn)
            outs_h.append(hn)
        
        
        out_h = torch.stack(outs_h, dim=1)
        
        out_h = self.fc_h2(self.fc_h1(out_h))
        # print('out_c', out_c.shape)
        out_d = torch.stack(outs_d, dim=1)
        out_d = self.fc_d2(self.fc_d1(out_d))
        # print('out_h', out_h.shape)
        # out_h = torch.tanh(out_h)
        # print('out_h', out_h.shape)
        out = torch.sum(out_h * out_d, dim=1)
        # print('out', out)
            
    

        # out_h = outs_h[-1].squeeze()
        
        # out = self.fc(out) 
        # out.size() --> 100, 10
        return out.squeeze()
 
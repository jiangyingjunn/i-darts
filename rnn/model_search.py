import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from torch.autograd import Variable
from collections import namedtuple
from model import DARTSCell, RNNModel


class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    #probs = F.softmax(self.weights, dim=-1)

    offset = 0
    states = s0.unsqueeze(0)
    for i in range(STEPS):
      #node_probs = F.softmax(self.node_weights[i, 0:i+1], dim=-1)
      global_probs = F.softmax(self.weights[offset:offset + i + 1].view(-1), dim=-1).view(i + 1, len(PRIMITIVES))
      if self.training:
        masked_states = states * h_mask.unsqueeze(0)
      else:
        masked_states = states
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid)
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()

      s = torch.zeros_like(s0)
      for k, name in enumerate(PRIMITIVES):
        if name == 'none':
          continue
        fn = self._get_activation(name)
        unweighted = states + c * (fn(h) - states)
        #s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
        s += torch.sum(global_probs[0:0+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
        # s += torch.sum((node_probs * probs[offset:offset + i + 1, k]).unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
      s = self.bn(s)
      states = torch.cat([states, s.unsqueeze(0)], 0)
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0)
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
      k = sum(i for i in range(1, STEPS+1))
      weights_data = torch.randn(k, len(PRIMITIVES)).mul_(1e-3)
      #node_weights_data = torch.randn(STEPS, STEPS).mul_(1e-3)
      self.weights = Variable(weights_data.cuda(), requires_grad=True)
      #self.node_weights = Variable(node_weights_data.cuda(), requires_grad = True)
      #self._arch_parameters = [self.weights, self.node_weights]
      self._arch_parameters = [self.weights]
      for rnn in self.rnns:
        rnn.weights = self.weights
        #rnn.node_weights = self.node_weights

    def arch_parameters(self):
      return self._arch_parameters

    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
      return loss, hidden_next

    def genotype(self):

      def _parse(probs):
        gene = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W = probs[start:end].copy()
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          start = end
        return gene

      def _parse_global(probs):
        gene = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W = F.softmax(probs[start:end].view(-1), dim=-1).view(i + 1, len(PRIMITIVES)).data.cpu().numpy()
          # W = probs[start:end].copy()
          listj = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))
          j = listj[0]
          if j==0 and i>0:
            j = listj[1]
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          start = end
        return gene
      def _parse_node(probs, node_probs):
        gene = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W_node = F.softmax(node_probs[i, 0:i + 1], dim=-1).data.cpu().numpy()
          W = probs[start:end].copy()
          j_best = None
          k_best = None

          for j in range(len(W_node)):
            if j_best is None or W_node[j] > W_node[j_best]:
              j_best = j

          for k in range(len(W[j_best])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j_best][k] > W[j_best][k_best]:
                k_best = k

          gene.append((PRIMITIVES[k_best], j_best))
          start = end
        return gene


      #gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
      gene = _parse_global(self.weights)
      #gene = _parse_node(F.softmax(self.weights, dim=-1).data.cpu().numpy(), self.node_weights)
      genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
      return genotype


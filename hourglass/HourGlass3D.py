import torch
import torch.nn as nn
from .Layers3D import *

def help(x):
        print(x.std(dim=2).mean())


class Hourglass3D(nn.Module):
	"""docstring for Hourglass3D"""
	def __init__(self, nChannels = 128, numReductions = 4, nModules = 2, poolKernel = (1,2,2), poolStride = (1,2,2), upSampleKernel = 2, temporal=-1):
		super(Hourglass3D, self).__init__()
		self.numReductions = numReductions
		self.nModules = nModules
		self.nChannels = nChannels
		self.poolKernel = poolKernel
		self.poolStride = poolStride
		self.upSampleKernel = upSampleKernel
		"""
		For the skip connection, a residual3D module (or sequence of residuaql modules)
		"""

		_skip = []
		for i in range(self.nModules):
			_skip.append(Residual3D(self.nChannels, self.nChannels, temporal[0][i]))

		self.skip = nn.Sequential(*_skip)
		
		"""
		First pooling to go to smaller dimension then pass input through 
		Residual Module or sequence of Modules then  and subsequent cases:
			either pass through Hourglass3D of numReductions-1
			or pass through Residual3D Module or sequence of Modules
		"""

		self.mp = nn.MaxPool3d(self.poolKernel, self.poolStride)
		
		_afterpool = []
		for i in range(self.nModules):
			_afterpool.append(Residual3D(self.nChannels, self.nChannels, temporal[1][i]))

		self.afterpool = nn.Sequential(*_afterpool)	

		if (numReductions > 1):
			self.hg = Hourglass3D(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride, self.upSampleKernel, temporal[2])
		else:
			_num1res = []
			for i in range(self.nModules):
				_num1res.append(Residual3D(self.nChannels,self.nChannels, temporal[2][i]))
			
			self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?
		
		"""
		Now another Residual3D Module or sequence of Residual3D Modules
		"""
		
		_lowres = []
		for i in range(self.nModules):
			_lowres.append(Residual3D(self.nChannels,self.nChannels, temporal[3][i]))

		self.lowres = nn.Sequential(*_lowres)

		"""
		Upsampling Layer (Can we change this??????)  
		As per Newell's paper upsamping recommended
		"""
		self.up = nn.Upsample(scale_factor = self.upSampleKernel)
		
		"""
		If temporal dimension is odd then after upsampling add a dimension temporally
		doing this via 2 kernel 1D convolution with 1 padding along the temporal direction
		"""
		#self.addTemporal = nn.ReplicationPad3d((0,0,0,0,0,1))

	def forward(self, input):
		out1 = input
		#help(out1)
		out1 = self.skip(out1)
		#print('skip %d'%(self.numReductions))
		#help(out1)
		out2 = input
		
		out2 = self.mp(out2)
		
		out2 = self.afterpool(out2)
		#print('out2 %d'%(self.numReductions))
		#help(out2)
		if self.numReductions>1:
			out2 = self.hg(out2)
		else:
			out2 = self.num1res(out2)
		#help(out2)
		out2 = self.lowres(out2)
		#help(out2)	
		
		N,C,D,H,W = out2.size()
		out2 = out2.transpose(1,2).contiguous().view(N*D,C,H,W).contiguous()
		out2 = self.up(out2)
		N1,C1,H1,W1 = out2.size()
		out2 = out2.view(N,D,C1,H1,W1).contiguous().transpose(1,2).contiguous()
		return out2 + out1	
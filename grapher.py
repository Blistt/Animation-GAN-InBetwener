import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", input_channels * 2, hidden_channels * 2, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
    to_BatchNorm("batchnorm1", offset="(0,0,0)", to="(conv1-east)"),
    to_LeakyReLU("leakyrelu1", offset="(0,0,0)", to="(batchnorm1-east)"),
    to_Conv("conv2", hidden_channels * 2, hidden_channels * 2, offset="(1,0,0)", to="(leakyrelu1-east)", height=32, depth=32, width=2 ),
    to_connection( "leakyrelu1", "conv2"), 
    to_BatchNorm("batchnorm2", offset="(0,0,0)", to="(conv2-east)"),
    to_LeakyReLU("leakyrelu2", offset="(0,0,0)", to="(batchnorm2-east)"),
    to_Conv("conv3", hidden_channels * 4, hidden_channels * 4, offset="(1,0,0)", to="(leakyrelu2-east)", height=32, depth=32, width=2 ),
    to_connection( "leakyrelu2", "conv3"), 
    to_BatchNorm("batchnorm3", offset="(0,0,0)", to="(conv3-east)"),
    to_LeakyReLU("leakyrelu3", offset="(0,0,0)", to="(batchnorm3-east)"),
    to_Conv("conv4", hidden_channels * 8, hidden_channels * 8, offset="(1,0,0)", to="(leakyrelu3-east)", height=32, depth=32, width=2 ),
    to_connection( "leakyrelu3", "conv4"), 
    to_BatchNorm("batchnorm4", offset="(0,0,0)", to="(conv4-east)"),
    to_LeakyReLU("leakyrelu4", offset="(0,0,0)", to="(batchnorm4-east)"),
    to_ConvTranspose("convtranspose1", hidden_channels * 8, hidden_channels * 4, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
    to_BatchNorm("batchnorm1", offset="(0,0,0)", to="(convtranspose1-east)"),
    to_LeakyReLU("leakyrelu1", offset="(0,0,0)", to="(batchnorm1-east)"),
    to_ConvTranspose("convtranspose2", hidden_channels * 4, hidden_channels * 2, offset="(1,0,0)", to="(leakyrelu1-east)", height=32, depth=32, width=2 ),
    to_connection( "leakyrelu1", "convtranspose2"), 
    to_BatchNorm("batchnorm2", offset="(0,0,0)", to="(convtranspose2-east)"),
    to_LeakyReLU("leakyrelu2", offset="(0,0,0)", to="(batchnorm2-east)"),
    to_ConvTranspose("convtranspose3", hidden_channels * 2, hidden_channels * 1, offset="(1,0,0)", to="(leakyrelu2-east)", height=32, depth=32, width=2 ),
    to_connection( "leakyrelu2", "convtranspose3"), 
    to_BatchNorm("batchnorm3", offset="(0,0,0)", to="(convtranspose3-east)"),
    to_LeakyReLU("leakyrelu3", offset="(0,0,0)", to="(batchnorm3-east)"),
    to_ConvTranspose("convtranspose4", hidden_channels * 1, input_channels * 1, offset="(1,0,0)", to="(leakyrelu3-east)", height=32, depth=32,width=2 ),
    to_connection( "leakyrelu3", "convtranspose4"), 
    to_Sigmoid("sigmoid1", offset="(1.5,0,0)",to="(convtranspose4-east)"),
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
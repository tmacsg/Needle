import sys
sys.path.append('./python')
sys.path.append('./apps')

from tests.test_sequence_models import *

_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

if __name__ == '__main__':
    test_rnn(SEQ_LENGTHS[0], NUM_LAYERS[0], BATCH_SIZES[0], INPUT_SIZES[0], 
        HIDDEN_SIZES[0], BIAS[0], INIT_HIDDEN[0], NONLINEARITIES[0], _DEVICES[0])

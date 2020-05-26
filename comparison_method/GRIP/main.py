print("Starting test")
from def_train_eval import *
from data_stream import *
import pickle              # import module first

DATA = 'NEW_LYFT'
SUFIX = 'trial_lstm_10_4_0.001'  #shorter is 10 epochs and batch size is 10 lr=0.1
TRAIN = False
EVAL = True
DIR = '../../../resources/data/{}/'.format(DATA)

epochs = 1
save_per_epoch = 1


train_seq_len = 6
pred_seq_len = 10


if __name__ == "__main__":

    if TRAIN:
        f2 = open ( DIR + 'stream2_obs_data_train.pkl','rb')  # 'r' for reading; can be omitted
        g2 = open ( DIR + 'stream2_pred_data_train.pkl', 'rb')  # 'r' for reading; can be omitted
        tr_seq_2 = pickle.load ( f2 )  # load file content as mydict
        pred_seq_2 = pickle.load ( g2 )  # load file content as mydict

        f2.close ()
        g2.close ()


        encoder, decoder, grip = trainIters(epochs, tr_seq_2 , pred_seq_2, DATA, SUFIX, print_every=1, save_every=save_per_epoch)

    if EVAL:
        print('start evaluating...')
        f2 = open ( DIR + 'stream2_obs_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
        g2 = open ( DIR + 'stream2_pred_data_test.pkl', 'rb')  # 'r' for reading; can be omitted
        tr_seq_2 = pickle.load ( f2 )  # load file content as mydict
        pred_seq_2 = pickle.load ( g2 )  # load file content as mydict

        eval(1, tr_seq_2, pred_seq_2, DATA, SUFIX)






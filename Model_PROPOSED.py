from Evaluation import evaluation
from Model_LSTM import Model_LSTM
from Model_TCN import Model_TCN


def Model_PROPOSED(Train_Data, Train_Target, Test_data, Test_Target):
    Eval1, pred1 = Model_LSTM(Train_Data, Train_Target, Test_data, Test_Target)
    Eval2, pred2 = Model_TCN(Train_Data, Train_Target, Test_data, Test_Target)
    pred = ((pred1 + pred2) / 2).astype('int')
    Eval = evaluation(pred, Test_Target)
    return pred, Eval

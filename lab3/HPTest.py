class LyrParams:
        num_channels=0
        dropout_rate=0.0
        regularization=None
        activation_function=''
        non_train_layer=''
        def __init__(self,num_channels,dropout_rate,regularization):
                self.num_channels=num_channels
                self.dropout_rate=dropout_rate
                self.regularization=regularization
        def __str__(self):
                return str(self.num_channels)+" channels "+str(self.dropout_rate)+" reg "+self.regularization
x=LyrParams(32,0.5,"L1")
print(x)

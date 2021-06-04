# Semi-Supervised Sequence Learning

1. Uses 2 approaches to improve sequence learning with recurrent neural networks
	a. Predict next word --> Causal Language Modeling
	b. Use a sequence autoencoder which reads the input sequence into a hidden vector and tries to predict the input sentence again

2. In their experiments, pretraining on unlabeled data from related tasks can improve generalization of subsequent supervised model --> To be tested on DistilBert

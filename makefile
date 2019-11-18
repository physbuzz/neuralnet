neuralnet: neuralnet.cpp
	g++ -o neuralnet -g neuralnet.cpp

run: neuralnet
	./neuralnet	

clean:
	-rm neuralnet

train: neuralnet
	./neuralnet | tee out.txt


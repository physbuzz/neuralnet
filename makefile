neuralnet: neuralnet.cpp
	g++ -o neuralnet -O3 neuralnet.cpp

run: neuralnet
	./neuralnet	

clean:
	-rm neuralnet

train: neuralnet
	./neuralnet | tee out.txt


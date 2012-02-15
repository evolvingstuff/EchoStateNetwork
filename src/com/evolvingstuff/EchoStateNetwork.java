package com.evolvingstuff;

import java.util.Random;

public class EchoStateNetwork {
	EchoStateLiquid liquid;
	NeuralNetwork readout;
		
	public EchoStateNetwork(Random r, int input_dimension, int liquid_hidden_dimension, int readout_hidden_dimension, int output_dimension) throws Exception {
		liquid = new EchoStateLiquid(r, input_dimension, liquid_hidden_dimension);
		readout = NeuralNetwork.FactoryFeedForwardNeuralNetwork(r, liquid_hidden_dimension, readout_hidden_dimension, output_dimension, NeuronType.Sigmoid);
		liquid.Reset();
	}
	
	public void Reset() {
		liquid.Reset();
	}
	
	public double[] Next(double[] input, double[] target_output) throws Exception {
		double[] liquid_state = liquid.Next(input);
		double[] output = readout.Next(liquid_state, target_output);
		return output;
	}

	public double[] Next(double[] input) throws Exception {
		double[] liquid_state = liquid.Next(input);
		double[] output = readout.Next(liquid_state);
		return output;
	}
	
	public void ShowState() {
		liquid.ShowState();
	}

}

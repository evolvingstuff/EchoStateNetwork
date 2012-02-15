package com.evolvingstuff;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class NeuralNetwork {
	
	enum CreationMethod {
		FeedForward,
		Recurrent
	}

	
	public static void Save(NeuralNetwork nn, String filepath) throws IOException {
		FileWriter fw = new FileWriter(new File(filepath));
		fw.write(nn.creationMethod.toString() + "\n");
		fw.write(nn.input_dimension + "\n");
		fw.write(nn.hidden_dimension + "\n");
		fw.write(nn.output_dimension + "\n");
		fw.write(nn.hiddenNeuronType.toString() + "\n");
		double[] params = nn.GetParameters();
		for (int d = 0; d < params.length; d++)
			fw.write(params[d] + " ");
		fw.flush();
		fw.close();
	}
	
	public static NeuralNetwork Load(String filepath) throws Exception {
		Scanner sc = new Scanner(new File(filepath));
		List<String> lines = new ArrayList<String>();
		while (sc.hasNextLine()) {
			String line = sc.nextLine();
			lines.add(line);
		}
		CreationMethod creationMethod = CreationMethod.valueOf(lines.get(0));
		int inputDimension = Integer.valueOf(lines.get(1));
		int hiddenDimension = Integer.valueOf(lines.get(2));
		int outputDimension = Integer.valueOf(lines.get(3));
		NeuronType neuronType = NeuronType.valueOf(lines.get(4));
		
		NeuralNetwork result = null;
		Random r = new Random();
		
		if (creationMethod == CreationMethod.FeedForward) {
			result = NeuralNetwork.FactoryFeedForwardNeuralNetwork(r, inputDimension, hiddenDimension, outputDimension, neuronType);
		}
		else if (creationMethod == CreationMethod.Recurrent) {
			result = NeuralNetwork.FactoryFeedForwardNeuralNetwork(r, inputDimension, hiddenDimension, outputDimension, neuronType);
		}
		else
			throw new Exception("Unknown creation method.");
		
		int totParams = result.GetParameters().length;
		String[] parts = lines.get(5).split(" ");
		double[] params = new double[totParams];
		for (int d = 0; d < totParams; d++) {
			params[d] = Double.valueOf(parts[d]);
		}
		result.SetParameters(params);
		return result;
	}
	
	class Node {
		
		Node(Random r, Neuron neuron, String name) {
			this.neuron = neuron;
			this.name = name;
			this.bias = r.nextGaussian() * INIT_WEIGHT_RANGE;
		}
		
		void Display() {
			System.out.println(name + " (sum=" + summation + " act="+activation+" pre-act="+prev_activation+" pre-delta="+pre_delta+" post-delta="+post_delta+")");
			System.out.println("\tbias = " + bias);
			for (Link link : instar_links) {
				if (link.recurrent)
					System.out.println("\t" + link.pre_node.name + " -> " + link.weight + " (recurrent)");
				else
					System.out.println("\t" + link.pre_node.name + " -> " + link.weight);
			}
		}
		
		void Reset() {
			prev_activation = 0;
			summation = 0;
			activation = 0;
			post_delta = 0;
			pre_delta = 0;
		}
		
		void AddInput(Node pre_node, Random r) {
			Link link = new Link(pre_node, this, r, false);
			instar_links.add(link);
		}
		
		void AddDelayedInput(Node pre_node, Random r) {
			Link link = new Link(pre_node, this, r, true);
			instar_links.add(link);
		}
		
		void Forwardprop() {
			prev_activation = activation; //rollover activation
			summation = 0;
			for (Link link : instar_links) {
				if (link.recurrent) 
					summation += link.pre_node.prev_activation * link.weight;
				else
					summation += link.pre_node.activation * link.weight;
			}
			summation += bias;
			activation = neuron.Activate(summation);
		}
		
		void Backprop() {
			pre_delta = neuron.Derivative(summation) * post_delta;
			for (Link link : instar_links) {
				if (link.recurrent) {
					//Truncated gradients
					link.weight += link.pre_node.prev_activation * pre_delta * LEARNING_RATE;
				}
				else {
					link.pre_node.post_delta += pre_delta * link.weight;
					link.weight += link.pre_node.activation * pre_delta * LEARNING_RATE;	
				}
			}
			bias += pre_delta * LEARNING_RATE;
		}
		
		List<Link> instar_links = new ArrayList<Link>();
		double bias;
		double summation;
		double activation;
		double post_delta;
		double pre_delta;
		String name;
		Neuron neuron;
		double prev_activation;
	}
	
	class Link {
		
		Link(Node pre_node, Node post_node, Random r, boolean recurrent) {
			this.pre_node = pre_node;
			this.post_node = post_node;
			this.weight = r.nextGaussian() * INIT_WEIGHT_RANGE;
			this.recurrent = recurrent;
		}
		
		Node pre_node;
		Node post_node;
		double weight;
		boolean recurrent = false;
	}
	
	CreationMethod creationMethod;
	NeuronType hiddenNeuronType;
	List<Node> input_nodes = new ArrayList<Node>();
	List<Node> hidden_nodes = new ArrayList<Node>();
	List<Node> output_nodes = new ArrayList<Node>();
	int input_dimension;
	int hidden_dimension;
	int output_dimension;
	public static double OUTPUT_GAIN = 1.0;
	public static double INIT_WEIGHT_RANGE = 0.1;
	public static double LEARNING_RATE = 0.1;
	double[] input_bias;
	
	public static NeuralNetwork FactoryFeedForwardNeuralNetwork(Random r, int input_dimension, int hidden_dimension, int output_dimension, NeuronType hidden_neuron_type) {
		NeuralNetwork result = new NeuralNetwork(r, input_dimension, hidden_dimension, output_dimension, hidden_neuron_type, CreationMethod.FeedForward);

		//Connect inputs
		for (Node pre_node : result.input_nodes) {
			//Input to Hidden
			for (Node post_node : result.hidden_nodes)
				post_node.AddInput(pre_node, r);
		}
		
		//Connect hidden
		for (int h = 0; h < hidden_dimension; h++) {
			//Hidden to Output
			for (Node output_node : result.output_nodes)
				output_node.AddInput(result.hidden_nodes.get(h), r);
		}

		return result;
	}
	
	public static NeuralNetwork FactoryRecurrentNeuralNetwork(Random r, int input_dimension, int hidden_dimension, int output_dimension, NeuronType hidden_neuron_type) {
		NeuralNetwork result = new NeuralNetwork(r, input_dimension, hidden_dimension, output_dimension, hidden_neuron_type, CreationMethod.Recurrent);
		
		//Connect inputs
		for (Node pre_node : result.input_nodes) {
			//Input to Hidden
			for (Node post_node : result.hidden_nodes)
				post_node.AddInput(pre_node, r);
		}
		
		//Connect recurrent
		for (Node post_hidden_node : result.hidden_nodes) {
			for (Node pre_hidden_node : result.hidden_nodes) {
				post_hidden_node.AddDelayedInput(pre_hidden_node, r);
			}
		}
		
		//Hidden to Output
		for (int h2 = 0; h2 < hidden_dimension; h2++) {
			Node post_hidden_node = result.hidden_nodes.get(h2);
			for (Node output_node : result.output_nodes)
				output_node.AddInput(post_hidden_node, r);
		}

		return result;
	}
	
	private NeuralNetwork(Random r, int input_dimension, int hidden_dimension, int output_dimension, NeuronType hidden_neuron_type, CreationMethod creationMethod) {
		this.input_dimension = input_dimension;
		this.hidden_dimension = hidden_dimension;
		this.output_dimension = output_dimension;
		for (int i = 0; i < input_dimension; i++)
			input_nodes.add(new Node(r, Neuron.Factory(NeuronType.Identity),"INPUT"+i) );
		for (int j = 0; j < hidden_dimension; j++)
			hidden_nodes.add(new Node(r, Neuron.Factory(hidden_neuron_type), "HIDDEN"+j));
		for (int k = 0; k < output_dimension; k++)
			output_nodes.add(new Node(r, Neuron.Factory(NeuronType.Identity),"OUTPUT"+k));
		this.creationMethod = creationMethod;
		this.hiddenNeuronType = hidden_neuron_type;
	}

	public double[] GetParameters() {
		List<Double> params = new ArrayList<Double>();
		for (Node hidden_node : hidden_nodes) {
			for (Link link : hidden_node.instar_links)
				params.add(link.weight);
			params.add(hidden_node.bias);
		}
		for (Node output_node : output_nodes) {
			for (Link link : output_node.instar_links)
				params.add(link.weight);
			params.add(output_node.bias);
		}
		double[] result = new double[params.size()];
		for (int i = 0; i < params.size(); i++)
			result[i] = params.get(i);
		return result;
	}

	public void SetParameters(double[] params) {
		int loc = 0;
		for (Node hidden_node : hidden_nodes) {
			for (Link link : hidden_node.instar_links)
				link.weight = params[loc++];
			hidden_node.bias = params[loc++];
		}
		for (Node output_node : output_nodes) {
			for (Link link : output_node.instar_links)
				link.weight = params[loc++];
			output_node.bias = params[loc++];
		}
	}
	
	public void Display() {
		for (Node hidden_node : hidden_nodes)
			hidden_node.Display();
		for (Node output_node : output_nodes)
			output_node.Display();
	}
	
	public void Reset() {
		for (Node hidden_node : hidden_nodes)
			hidden_node.Reset();
	}
	
	public double[] Next(double[] input) throws Exception {
		for (int i = 0; i < input_dimension; i++)
			input_nodes.get(i).activation = input[i];
		for (Node hidden_node : hidden_nodes)
			hidden_node.Forwardprop();
		for (Node output_node : output_nodes)
			output_node.Forwardprop();
		double[] result = new double[output_dimension];
		for (int k = 0; k < output_dimension; k++)
			result[k] = output_nodes.get(k).activation;
		return result;
	}

	public double[] Next(double[] input, double[] target_output) throws Exception {
		double[] result = new double[output_dimension];
		for (int i = 0; i < input_dimension; i++) {
			input_nodes.get(i).activation = input[i];
			input_nodes.get(i).post_delta = 0;
		}
		for (Node hidden_node : hidden_nodes) {
			hidden_node.Forwardprop();
			hidden_node.post_delta = 0;
		}
		for (int k = 0; k < output_dimension; k++) {
			output_nodes.get(k).Forwardprop();
			result[k] = output_nodes.get(k).activation;
			output_nodes.get(k).post_delta = OUTPUT_GAIN * (target_output[k] - result[k]); //TODO: test this!
			output_nodes.get(k).Backprop();
		}

		for (int j = hidden_dimension - 1; j >= 0; j--) // MUST DO IN REVERSE ORDER!!
			hidden_nodes.get(j).Backprop();
		return result;
	}
	
	public int GetInputDimension() {
		return this.input_dimension;
	}

}

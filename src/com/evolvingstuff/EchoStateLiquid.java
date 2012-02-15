package com.evolvingstuff;

import java.util.Random;

import com.evolvingstuff.Neuron;
import com.evolvingstuff.NeuronType;
import Jama.Matrix;

public class EchoStateLiquid {

	private double[][] IH;
	private double[][] HH;
	private double[] context;
	private int input_dimension;
	private int hidden_dimension;
	private double input_weight_range = 1.0;
	private double spectral_radius = 0.8;
	private Neuron neuron;
	private NeuronType neuron_type = NeuronType.Tanh;
	
	private void InitWeights(Random r) throws Exception {
		context = new double[hidden_dimension];
		IH = new double[hidden_dimension][input_dimension];
		HH = new double[hidden_dimension][hidden_dimension];
		
		for (int j = 0; j < hidden_dimension; j++) {
			for (int i = 0; i < input_dimension; i++)
				IH[j][i] = r.nextGaussian() * (input_weight_range/(double)input_dimension);	
			for (int k = 0; k < hidden_dimension; k++)
				HH[k][j] = r.nextGaussian();
		}
		
		//rescale for spectral radius
		Matrix m = new Matrix(HH);
		double[] eigs = m.eig().getRealEigenvalues();
		double mx = Double.NEGATIVE_INFINITY;
		for (int e = 0; e < eigs.length; e++) {
			if (Math.abs(eigs[e]) > mx)
				mx = Math.abs(eigs[e]);
		}
		
		for (int j = 0; j < hidden_dimension; j++) {
			for (int k = 0; k < hidden_dimension; k++)
				HH[k][j] *= spectral_radius / mx;
		}
		
		//test
		m = new Matrix(HH);
		eigs = m.eig().getRealEigenvalues();
		mx = Double.NEGATIVE_INFINITY;
		for (int e = 0; e < eigs.length; e++) {
			if (Math.abs(eigs[e]) > mx)
				mx = Math.abs(eigs[e]);
		}
		
		if (Math.abs(mx - spectral_radius) > 0.01) 
			throw new Exception("Unexpected spectral radius");
	}
	
	public EchoStateLiquid(Random r, int input_dimension, int hidden_dimension) throws Exception{
		
		this.input_dimension = input_dimension;
		this.hidden_dimension = hidden_dimension;
		this.neuron = Neuron.Factory(neuron_type);
		
		InitWeights(r);
	}
	
	public EchoStateLiquid(Random r, int input_dimension, int hidden_dimension, double spectral_radius) throws Exception{
		
		this.input_dimension = input_dimension;
		this.hidden_dimension = hidden_dimension;
		this.neuron = Neuron.Factory(neuron_type);
		this.spectral_radius = spectral_radius;
		
		InitWeights(r);
	}
	
	public void Reset() {
		for (int j = 0; j < hidden_dimension; j++)
			context[j] = 0;
	}

	public double[] Next(double[] input) throws Exception {
		double[] accum = new double[hidden_dimension];
		double[] result = new double[hidden_dimension];
		for (int j = 0; j < hidden_dimension; j++) {
			for (int i = 0; i < input_dimension; i++)
				accum[j] += IH[j][i] * input[i];
			for (int k = 0; k < hidden_dimension; k++)
				accum[j] += HH[k][j] * context[j];
		}
		for (int j = 0; j < hidden_dimension; j++) {
			context[j] = neuron.Activate(accum[j]);
			result[j] = context[j];
		}
		return result;
	}

}

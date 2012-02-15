import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import com.evolvingstuff.EchoStateNetwork;
import com.evolvingstuff.NeuralNetwork;

public class Test {
	public static void main(String[] args) throws Exception {
		System.out.println("Test of Echo State Network");
		List<Double> data = new ArrayList<Double>();
		Scanner sc = new Scanner(new File("mg30.dat"));
		while (sc.hasNext())
			data.add(Double.parseDouble(sc.nextLine().trim()));
		Random r = new Random();
		int input_dimension = 1;
		int output_dimension = 1;
		int liquid_dimension = 100;
		int readout_dimension = 10;
		NeuralNetwork.LEARNING_RATE = 0.1;
		EchoStateNetwork esn = new EchoStateNetwork(r, input_dimension, liquid_dimension, readout_dimension, output_dimension);
		int washout_period = 99;
		int train_period = 900;
		int test_period = 500;
		
		System.out.println("total_examples = " + data.size());
		
		double[] input = new double[input_dimension];
		double[] output = new double[output_dimension];
		double[] target_output = new double[output_dimension];
		double error;
		
		for (int epoch = 0; epoch < 5000; epoch++) {
			esn.Reset();
			double tot_error_train = 0;
			double tot_error_test = 0;
			for (int t = 0; t < data.size()-1; t++) {
				input[0] = data.get(t);
				if (t < washout_period)
					esn.Next(input);
				else if (t < washout_period + train_period) {
					target_output[0] = data.get(t+1);
					output = esn.Next(input, target_output);
					error = Math.abs(target_output[0] - output[0]);
					tot_error_train += error;
				}
				else {
					target_output[0] = data.get(t+1);
					output = esn.Next(input);
					error = Math.abs(target_output[0] - output[0]);
					tot_error_test += error;
				}
				//esn.ShowState();
			}
			double avg_error_train = tot_error_train / train_period;
			double avg_error_test = tot_error_test / test_period;
			if (epoch % 10 == 9 || epoch == 0)
			System.out.println("["+(epoch + 1)+"]\tTrain error: " + avg_error_train + "\tTest error: " + avg_error_test);
		}
		System.out.println("done.");
	}

}

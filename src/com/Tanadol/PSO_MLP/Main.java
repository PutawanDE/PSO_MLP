package com.Tanadol.PSO_MLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    private static int[] nodes = new int[]{8, 4, 1};
    private static final MathFunction leakyReluFn = (x) -> {
        if (x <= 0) return 0.01 * x;
        else return x;
    };

    private static double minWeight = -1.0;
    private static double maxWeight = 1.0;

    private static double minV = -1.0;
    private static double maxV = 1.0;

    public static void main(String[] args) {

    }

    private static void train(Pair<double[][], double[][]> inOut, int k) {
        int rows = inOut.x.length;

        int numDimensions = 0;
        for (int i = 1; i <= nodes.length; i++) {
            numDimensions += nodes[i] * nodes[i - 1];
        }

        Swarm swarm = new Swarm(50, numDimensions,
                new double[]{minWeight, maxWeight}, new double[]{minV, maxV});

        for (int i = 0; i < rows; i++) {
            Matrix[] biases = new Matrix[nodes.length - 1];
            for (int j = 0; j < nodes.length - 1; j++) {
                biases[j] = new Matrix(nodes[j + 1], 1);
            }

            Network network = new Network(nodes, leakyReluFn, leakyReluFn, minWeight, maxWeight, biases);

            swarm.run(100, 0, 0, 0, network, inOut.x[i], inOut.y[i]);
        }
    }

    private static Pair<double[][], double[][]> readTrainingData(String filename) throws IOException {
        List<double[]> input = new ArrayList<>();
        List<double[]> output = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        while ((line = br.readLine()) != null) {
            String[] cols = line.split(",");
            if (cols[1].equals("M")) {
                output.add(new double[]{1.0});
            } else {
                output.add(new double[]{0.0});
            }

            double[] inputVect = new double[30];
            for (int j = 0; j < 30; j++) {
                inputVect[j] = Double.parseDouble(cols[j + 2]);
            }
            input.add(inputVect);
        }

        return new Pair<>(input.toArray(new double[0][0]), output.toArray(new double[0][0]));
    }
}

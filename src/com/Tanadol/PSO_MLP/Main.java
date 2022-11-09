package com.Tanadol.PSO_MLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    private static int[] nodes = new int[]{8, 5, 1};
    private static final MathFunction leakyReluFn = (x) -> {
        if (x <= 0) return 0.01 * x;
        else return x;
    };

    private static final MathFunction sigmoidFn = (x) -> 1.0 / (1.0 + Math.exp(x));

    private static final MathFunction tanhFn = (x) -> 2.0 / (1 + Math.exp(-2.0 * x)) - 1.0;


    private static double minWeight = -1.0;
    private static double maxWeight = 1.0;

    private static double minV = -1.0;
    private static double maxV = 1.0;

    public static void main(String[] args) throws IOException {
        int k = 10;

        List<Pair<double[][], double[][]>> folds =
                readSplitTrainingData("D:\\PUTAWAN\\ComputerProjects\\CI\\HW4-PSO\\Data\\AirQualityUCI_shuffle.csv", k, 9300);

        int rows = folds.get(0).x.length;

        int numDimensions = 0;
        for (int j = 1; j < nodes.length; j++) {
            numDimensions += nodes[j] * nodes[j - 1];
        }

        for (int i = 0; i < k; i++) {
            List<Pair<double[][], double[][]>> normData = minMaxNorm(folds, i);

            Matrix[] biases = new Matrix[nodes.length - 1];
            for (int j = 0; j < nodes.length - 1; j++) {
                biases[j] = new Matrix(nodes[j + 1], 1);
            }

            Swarm swarm = new Swarm(50, numDimensions,
                    new double[]{minWeight, maxWeight}, new double[]{minV, maxV});

            Network network = new Network(nodes, leakyReluFn, leakyReluFn, minWeight, maxWeight, biases);

            for (int j = 0; j < normData.size(); j++) {
                if (j == i) continue;

                for (int r = 0; r < rows; r++) {
                    swarm.run(100, 0.5, 0.7, 0.4, network, normData.get(j).x[r], normData.get(j).y[r]);
                }
            }

            Matrix bestSolution = swarm.getBestSolution();
            network.setWeights(bestSolution.data[0]);
            for (int r = 0; r < rows; r++) {
                System.out.println("loss: " + network.feedForward(normData.get(i).x[r], normData.get(i).y[r]));
                System.out.println("output: " + network.activations[nodes.length - 1].data[0][0]);
            }
        }
    }

    private static List<Pair<double[][], double[][]>> readSplitTrainingData(String filename, int k, int sampleSize) throws IOException {
        List<double[]> input = new ArrayList<>();
        List<double[]> output = new ArrayList<>();
        ArrayList<Pair<double[][], double[][]>> folds = new ArrayList<>(k);

        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        int f = 1;
        int l = 1;
        int foldSize = sampleSize / k;
        while ((line = br.readLine()) != null) {
            if (l > sampleSize) break;

            String[] cols = line.split(",");
            input.add(new double[]{
                    Double.parseDouble(cols[2]),
                    Double.parseDouble(cols[5]),
                    Double.parseDouble(cols[7]),
                    Double.parseDouble(cols[9]),
                    Double.parseDouble(cols[10]),
                    Double.parseDouble(cols[11]),
                    Double.parseDouble(cols[12]),
                    Double.parseDouble(cols[13])
            });
            output.add(new double[]{Double.parseDouble(cols[4])});

            if (l == f * foldSize) {
                folds.add(new Pair<>(input.toArray(new double[0][0]), output.toArray(new double[0][0])));
                input.clear();
                output.clear();
                f++;
            }

            l++;
        }

        return folds;
    }

    private static List<Pair<double[][], double[][]>> minMaxNorm(List<Pair<double[][], double[][]>> trainTestData,
                                                                 int testSetIdx) {
        // clone data
        List<Pair<double[][], double[][]>> data = new ArrayList<>(trainTestData.size());
        for (Pair<double[][], double[][]> part : trainTestData) {
            double[][] input = copy2Darray(part.x, 0, part.x[0].length);
            double[][] output = copy2Darray(part.y, 0, part.y[0].length);

            data.add(new Pair<>(input, output));
        }


        double[] min = new double[data.get(0).x[0].length + 1];
        double[] range = new double[data.get(0).x[0].length + 1];

        Arrays.fill(min, Double.POSITIVE_INFINITY);
        double[] max = new double[data.get(0).x[0].length + 1];
        Arrays.fill(max, Double.NEGATIVE_INFINITY);

        // find min max
        for (int i = 0; i < data.size(); i++) {
            if (i == testSetIdx) continue;

            for (int j = 0; j < data.get(i).x.length; j++) {
                for (int k = 0; k < data.get(i).x[j].length; k++) {
                    if (data.get(i).x[j][k] <= min[k]) {
                        min[k] = data.get(i).x[j][k];
                    }

                    if (data.get(i).x[j][k] >= max[k]) {
                        max[k] = data.get(i).x[j][k];
                    }
                }

                if (data.get(i).y[j][0] > max[max.length - 1]) {
                    max[max.length - 1] = data.get(i).y[j][0];
                }

                if (data.get(i).y[j][0] < min[max.length - 1]) {
                    min[max.length - 1] = data.get(i).y[j][0];
                }
            }
        }

        for (int i = 0; i < max.length; i++) {
            range[i] = max[i] - min[i];
        }

        for (Pair<double[][], double[][]> datum : data) {
            for (int j = 0; j < datum.x.length; j++) {
                for (int k = 0; k < datum.x[j].length; k++) {
                    datum.x[j][k] = (datum.x[j][k] - min[k]) / range[k];
                }
                datum.y[j][0] = (datum.y[j][0] - min[max.length - 1]) / range[max.length - 1];
            }
        }

        return data;
    }

    protected static double[][] copy2Darray(double[][] src, int startCol, int colLen) {
        double[][] dataClone = new double[src.length][colLen];

        for (int i = 0; i < dataClone.length; i++) {
            double[] orgRow = src[i];
            System.arraycopy(orgRow, startCol, dataClone[i], 0, colLen);
        }
        return dataClone;
    }
}

package com.Tanadol.PSO_MLP;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

interface MathFunction {
    double run(double x);
}

public class Network {
    protected int inputLength;
    protected int desiredOutputLength;

    protected int layerCount;
    private int[] nodeInLayerCount;

    protected Matrix[] activations;
    protected Matrix[] weights;
    private Matrix[] biases;
    protected double loss;

    private double minWeight;
    private double maxWeight;

    private MathFunction hiddenLayerActivationFn;
    private MathFunction outputLayerActivationFn;

    private static final Random random = new Random();

    public Network(Network n) {
        this.inputLength = n.inputLength;
        this.desiredOutputLength = n.desiredOutputLength;

        this.layerCount = n.layerCount;
        this.nodeInLayerCount = n.nodeInLayerCount.clone();

        this.activations = new Matrix[n.activations.length];
        for (int i = 0; i < n.activations.length; i++) {
            this.activations[i] = new Matrix(n.activations[i]);
        }

        this.weights = new Matrix[n.weights.length];
        for (int i = 0; i < n.weights.length; i++) {
            this.weights[i] = new Matrix(n.weights[i]);
        }

        this.biases = new Matrix[n.biases.length];
        for (int i = 0; i < n.biases.length; i++) {
            this.biases[i] = new Matrix(n.biases[i]);
        }

        this.loss = n.loss;

        this.minWeight = n.minWeight;
        this.maxWeight = n.maxWeight;
        this.hiddenLayerActivationFn = n.hiddenLayerActivationFn;
        this.outputLayerActivationFn = n.hiddenLayerActivationFn;
    }

    public Network(int[] nodeInLayerCount, MathFunction hiddenLayerActivation,
                   MathFunction outputLayerActivation, double minWeight, double maxWeight, Matrix[] biases) {
        initNetwork(nodeInLayerCount, hiddenLayerActivation, outputLayerActivation, minWeight, maxWeight, biases);
        initRandomWeight();
    }

    public Network(int[] nodeInLayerCount, MathFunction hiddenLayerActivation,
                   MathFunction outputLayerActivation, double minWeight, double maxWeight, Matrix[] biases,
                   double[] weights) {
        initNetwork(nodeInLayerCount, hiddenLayerActivation, outputLayerActivation, minWeight, maxWeight, biases);

        // set custom weights
        setWeights(weights);
    }

    private void initNetwork(int[] nodeInLayerCount, MathFunction hiddenLayerActivation,
                             MathFunction outputLayerActivation, double minWeight, double maxWeight, Matrix[] biases) {
        this.layerCount = nodeInLayerCount.length;
        this.nodeInLayerCount = nodeInLayerCount;

        this.hiddenLayerActivationFn = hiddenLayerActivation;
        this.outputLayerActivationFn = outputLayerActivation;

        this.minWeight = minWeight;
        this.maxWeight = maxWeight;

        this.weights = new Matrix[layerCount - 1];
        this.biases = biases.clone();
        this.activations = new Matrix[layerCount];

        this.inputLength = nodeInLayerCount[0];
        this.desiredOutputLength = nodeInLayerCount[nodeInLayerCount.length - 1];
    }

    private void initRandomWeight() {
        for (int k = 0; k < weights.length; k++) {
            weights[k] = new Matrix(nodeInLayerCount[k + 1], nodeInLayerCount[k]);
            for (int j = 0; j < weights[k].getRows(); j++) {
                for (int i = 0; i < weights[k].getCols(); i++) {
                    weights[k].data[j][i] = random.nextDouble(minWeight, maxWeight);
                }
            }
        }
    }

    public void setWeights(double[] weights) {
        Matrix[] newWeights = new Matrix[nodeInLayerCount.length - 1];
        int k = 0;
        for (int l = 0; l < nodeInLayerCount.length - 1; l++) {
            newWeights[l] = new Matrix(nodeInLayerCount[l + 1], nodeInLayerCount[l]);
            for (int i = 0; i < newWeights[l].getRows(); i++) {
                for (int j = 0; j < newWeights[l].getCols(); j++) {
                    newWeights[l].data[i][j] = weights[k];
                    k++;
                }
            }
        }
        this.weights = newWeights;
    }

    protected double feedForward(double[] inputVect, double[] desiredOutputVect) {
        double[][] inputMat = new double[inputVect.length][1];
        for (int i = 0; i < inputVect.length; i++) {
            inputMat[i][0] = inputVect[i];
        }

        activations[0] = new Matrix(inputMat);
        for (int i = 1; i < layerCount; i++) {
            MathFunction activationFn = i == layerCount - 1 ? outputLayerActivationFn : hiddenLayerActivationFn;

            Matrix net = Matrix.multiply(weights[i - 1], activations[i - 1]);
            Matrix output = net.add(biases[i - 1]);
            activations[i] = Matrix.applyFunction(output, activationFn);
        }

        loss = calcLoss(desiredOutputVect);
        return loss;
    }

    // calculate Loss, MAE
    private double calcLoss(double[] desiredOutputVect) {
        double sumAbsErr = 0;

        for (int i = 0; i < desiredOutputLength; i++) {
            double error = desiredOutputVect[i] - activations[layerCount - 1].data[i][0];
            sumAbsErr = Math.abs(error);
        }
        loss = sumAbsErr / desiredOutputLength;
        return loss;
    }
}

package com.Tanadol.PSO_MLP;

import java.util.Arrays;
import java.util.Random;

public class Particle {
    private static final Random random = new Random();

    public Matrix x;
    public Matrix v;
    public Matrix vPrev;

    public double pbest= Double.POSITIVE_INFINITY;
    public double lbest = Double.POSITIVE_INFINITY;

    public Matrix x_pbest;
    public Matrix x_lbest;

    public Particle(int numDimensions, double[] xRange, double[] vRange) {
        x = new Matrix(1, numDimensions);
        v = new Matrix(1, numDimensions);
        vPrev = new Matrix(1, numDimensions);

        x_lbest = new Matrix(1, numDimensions);
        x_pbest = new Matrix(1, numDimensions);

        Arrays.setAll(x.data[0], i -> random.nextDouble(xRange[0], xRange[1]));
        Arrays.setAll(v.data[0], i -> random.nextDouble(vRange[0], vRange[1]));
    }
}

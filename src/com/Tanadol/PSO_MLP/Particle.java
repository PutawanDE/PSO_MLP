package com.Tanadol.PSO_MLP;

import java.util.Arrays;
import java.util.Random;

public class Particle {
    private static final Random random = new Random();

    public Matrix x;
    public Matrix v;
    public Matrix vPrev;

    public double pbest;
    public double lbest;

    public Matrix x_pbest;
    public Matrix x_lbest;

    public Particle(int numDimensions, double[] xRange, double[] vRange) {
        x = new Matrix(numDimensions, 1);
        v = new Matrix(numDimensions, 1);
        vPrev = new Matrix(numDimensions, 1);

        x_lbest = new Matrix(numDimensions, 1);
        x_pbest = new Matrix(numDimensions, 1);

        Arrays.setAll(x.data[0], i -> random.nextDouble(xRange[0], xRange[1]));
        Arrays.setAll(v.data[0], i -> random.nextDouble(vRange[0], vRange[1]));
        pbest = Double.POSITIVE_INFINITY;
    }
}

package com.Tanadol.PSO_MLP;

import java.util.Arrays;
import java.util.Random;

public class Swarm {
    private static final Random random = new Random();

    public Particle[] particles;
    public double gbest = Double.POSITIVE_INFINITY;
    public Matrix x_gbest;

    public Swarm(int numParticles, int numDimensions, double[] xRange, double[] vRange) {
        particles = new Particle[numParticles];
        Arrays.setAll(particles, i -> new Particle(numDimensions, xRange, vRange));
    }

    // lbest with Clerc and Kennedy Constriction and inertia weight
    public void run(int iteration, double c1, double c2, double inertiaWeight,
                    Network network, double[] input, double[] desiredOutput) {
        for (int i = 0; i < iteration; i++) {
            for (Particle p : particles) {
                network.setWeights(p.x.data[0]);
                double f = network.feedForward(input, desiredOutput);

                if (f < p.pbest) {
                    p.pbest = f;
                    p.x_pbest = new Matrix(p.x);
                }

                if (f < p.lbest) {
                    p.lbest = f;
                    p.x_lbest = new Matrix(p.x);
                }

                if (f < gbest) {
                    gbest = f;
                    x_gbest = new Matrix(p.x);
                }
            }

            for (Particle p : particles) {
                double rho1 = random.nextDouble() * c1;
                double rho2 = random.nextDouble() * c2;

                Matrix cognitive = p.x_pbest.subtract(p.x).multiplyByConstant(rho1);
                Matrix social = x_gbest.subtract(p.x).multiplyByConstant(rho2);

                p.v = p.v.multiplyByConstant(inertiaWeight).add(cognitive).add(social);

                p.x = p.x.add(p.v);
            }
        }
    }

    public Matrix getBestSolution() {
        return x_gbest;
    }
}

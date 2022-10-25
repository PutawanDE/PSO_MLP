package com.Tanadol.PSO_MLP;

import java.util.Arrays;
import java.util.Random;

public class Swarm {
    private static final Random random = new Random();

    public Particle[] particles;

    public Swarm(int numParticles, int numDimensions, double[] xRange, double[] vRange) {
        particles = new Particle[numParticles];
        Arrays.setAll(particles, i -> new Particle(numDimensions, xRange, vRange));
    }

    // lbest with Clerc and Kennedy Constriction and inertia weight
    public void run(int iteration, double c1, double c2, double inertiaWeight,
                    Network network, double[] input, double[] desiredOutput) {
        network = new Network(network);

        for (int i = 0; i < iteration; i++) {
            for (Particle p : particles) {
                network.setWeights(p.x.data[0]);
                double f = network.feedForward(input, desiredOutput);

                if (f < p.pbest) {
                    p.pbest = f;
                    p.x_lbest = new Matrix(p.x);
                }

                if (f < p.lbest) {
                    p.lbest = f;
                    p.x_lbest = new Matrix(p.x);
                }
            }

            for (Particle p : particles) {
                double rho1 = random.nextDouble() * c1;
                double rho2 = random.nextDouble() * c2;

                double constrictionCoeff = 1.0;
                if (rho1 + rho2 > 4.0) {
                    double rho = rho1 + rho2;
                    constrictionCoeff = 1.0 - (1.0 / rho) + Math.sqrt(Math.abs(rho * rho - 4 * rho));
                }

                Matrix cognitive = p.x_pbest.subtract(p.x).multiplyByConstant(rho1);
                Matrix social = p.x_lbest.subtract(p.x).multiplyByConstant(rho2);

                p.v = p.v.multiplyByConstant(inertiaWeight).add(cognitive).add(social)
                        .multiplyByConstant(constrictionCoeff);

                p.x = p.x.add(p.v);
            }
        }
    }
}

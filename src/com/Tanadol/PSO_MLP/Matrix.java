package com.Tanadol.PSO_MLP;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Matrix {
    private int rows;
    private int cols;

    public double[][] data;

    public Matrix(Matrix matrix) {
        this.rows = matrix.rows;
        this.cols = matrix.cols;

        if (matrix.data == null) {
            this.data = null;
        } else {
            double[][] cloneData = new double[matrix.data.length][];
            for (int i = 0; i < matrix.data.length; i++) {
                cloneData[i] = matrix.data[i].clone();
            }
            this.data = cloneData;
        }
    }

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.data = data.clone();
        this.rows = data.length;
        this.cols = data[0].length;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public Matrix add(Matrix matrix) {
        if (matrix.rows != rows && matrix.cols != cols) {
            throw new ArithmeticException("Matrix Addition is not possible.");
        }

        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] + matrix.data[i][j];
            }
        }

        return new Matrix(result);
    }

    public Matrix subtract(Matrix matrix) {
        if (matrix.rows != rows && matrix.cols != cols) {
            throw new ArithmeticException("Matrix Subtraction is not possible.");
        }

        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] - matrix.data[i][j];
            }
        }

        return new Matrix(result);
    }

    public Matrix multiplyByConstant(double c) {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] * c;
            }
        }

        return new Matrix(result);
    }

    public static Matrix multiply(Matrix a, Matrix b) {
        if (b.rows != a.cols) {
            throw new ArithmeticException("Matrix Multiplication is not possible.");
        }

        double[][] result = new double[a.rows][b.cols];
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < b.cols; j++) {
                for (int k = 0; k < b.rows; k++) {
                    result[i][j] += a.data[i][k] * b.data[k][j];
                }
            }
        }
        return new Matrix(result);
    }

    public static Matrix applyFunction(Matrix a, MathFunction fn) {
        double[][] result = new double[a.rows][a.cols];
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result[i][j] = fn.run(a.data[i][j]);
            }
        }
        return new Matrix(result);
    }

    @Override
    public String toString() {
        return Arrays
                .stream(data)
                .map(Arrays::toString)
                .collect(Collectors.joining(System.lineSeparator()));
    }
}

/*
 * Copyright 2021 German Vekhorev (DarksideCode)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package me.darksidecode.lvq4j;

import java.util.Objects;

/**
 * An interface used to normalize input data.
 * Normalization is one of the input data preprocession steps.
 *
 *     https://en.wikipedia.org/wiki/Feature_scaling
 *
 * This class file also contains some of the most widely used
 * standard input normalization functions that can be accessed
 * via "static" references.
 *
 * All of the default functions perform input validation,
 * and any custom implementation is ought to do so.
 * @see #validateInputVector(double[])
 */
public interface NormalizationFunction {

    /**
     * Validates and normalizes the specified input vector.
     * The last number in the array is the ID of a DataRecord
     * label and must not be used in the normalization algorithm
     * at all. It must be kept as is at the end of the array,
     * without affecting its other elements. For this reason,
     * "input.length - 1" is used whereever possible.
     *
     * @see #normalize(double[])
     *
     * @param input the vector whose values to normalize.
     */
    void normalize(double[] input);


    /*

    Below is the implementation of some of the most widely used algorithms.

     */


    /**
     * For every feature, the minimum value of that feature gets transformed into
     * a 0.0, the maximum value gets transformed into a 1.0, and every other value
     * gets transformed into a decimal between 0.0 and 1.0.
     *
     *     X' = (X - Xmin) / (Xmax - Xmin).
     *
     * Guarantees all features will have the exact same scale but does not handle outliers well.
     * @see #Z_SCORE if outliers handling is desired.
     */
    NormalizationFunction MIN_MAX = input -> {
        validateInputVector(input);

        double xMin = Double.MAX_VALUE;
        double xMax = Double.MIN_VALUE;

        for (int i = 0; i < input.length - 1; i++) {
            double x = input[i];

            xMin = Math.min(xMin, x);
            xMax = Math.max(xMax, x);
        }

        double d = xMax - xMin;

        for (int i = 0; i < input.length - 1; i++)
            input[i] = (input[i] - xMin) / d;
    };


    /**
     * Simple mean normalization.
     *
     *     X' = (X - Xaverage) / (Xmax - Xmin).
     *
     * There is another form of the means normalization which is when we divide
     * by the standard deviation which is also called standardization.
     * @see #Z_SCORE for a more widely used form of standartization.
     */
    NormalizationFunction MEAN = input -> {
        validateInputVector(input);

        double xMin = Double.MAX_VALUE;
        double xMax = Double.MIN_VALUE;
        double xAvg = 0.0;

        for (int i = 0; i < input.length - 1; i++) {
            double x = input[i];

            xMin = Math.min(xMin, x);
            xMax = Math.max(xMax, x);
            xAvg += x;
        }

        xAvg /= input.length - 1;
        double d = xMax - xMin;

        for (int i = 0; i < input.length - 1; i++)
            input[i] = (input[i] - xAvg) / d;
    };


    /**
     * If a value is exactly equal to the mean of all the values of the feature,
     * it will be normalized to 0.0. If it is below the mean, it will be a negative
     * number, and if it is above the mean it will be a positive number. The size
     * of those negative and positive numbers is determined by the standard deviation
     * of the original feature. If the unnormalized data had a large standard deviation,
     * the normalized values will be closer to 0.0. Most of the values will be in range
     * (-3z; 3z).
     *
     *     X' = (X - Xaverage) / z,
     *          where z is the standard deviation of the given feature vector.
     *
     * Handles outliers, but does not produce normalized data with the exact same scale.
     * @see #MIN_MAX if data scale accuracy is desired.
     * @see #MEAN for a simpler form of standartization.
     */
    NormalizationFunction Z_SCORE = input -> {
        validateInputVector(input);
        double xAvg = 0.0;

        for (int i = 0; i < input.length - 1; i++)
            xAvg += input[i];

        xAvg /= input.length - 1;
        double variance = 0.0;

        for (int i = 0; i < input.length - 1; i++) {
            double x = input[i];
            variance += (x - xAvg) * (x - xAvg);
        }

        variance /= input.length - 1;
        double stdDev = Math.sqrt(variance);

        for (int i = 0; i < input.length - 1; i++)
            input[i] = (input[i] - xAvg) / stdDev;
    };


    /**
     * Scaling the components of a feature vector such that the complete vector has length one.
     * In this case, this means dividing each component by the Euclidean length of the vector.
     *
     *     X' = X / ||X||.
     *
     * In some applications (e.g., histogram features) it can be more practical to use the L1 norm
     * (i.e., taxicab geometry) of the feature vector. This is especially important if in the
     * following learning steps the scalar metric is used as a distance measure.
     */
    NormalizationFunction UNIT_LEN_SCALE = input -> {
        validateInputVector(input);
        double squareSum = 0.0;
        
        for (int i = 0; i < input.length - 1; i++) {
            double x = input[i];
            squareSum += x * x;
        }

        double vecLen = Math.sqrt(squareSum);

        for (int i = 0; i < input.length - 1; i++)
            input[i] = input[i] / vecLen;
    };


    /**
     * Ensures that the specified input vector is safe to use in any calculation steps.
     *
     * @param input the vector to check.
     *
     * @throws NullPointerException if the specified vector is null.
     * @throws IllegalArgumentException if the specified vector is empty (of a zero length).
     * @throws ArithmeticException if the specified vector contains NaN or Infinite values.
     */
    static void validateInputVector(double[] input) {
        if (Objects.requireNonNull(input,
                "input vector cannot be null").length == 0)
            throw new IllegalArgumentException("input vector cannot be empty");

        for (int i = 0; i < input.length; i++) {
            double x = input[i];

            if (Double.isNaN(x))
                throw new ArithmeticException(
                        "input vector contains a NaN value at index " + i);

            if (Double.isInfinite(x))
                throw new ArithmeticException(
                        "input vector contains an Infinite value at index " + i);
        }
    }

}

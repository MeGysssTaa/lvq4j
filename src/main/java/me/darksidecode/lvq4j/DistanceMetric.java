/*
 * Copyright 2020 DarksideCode
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

/**
 * An interface used to measure the distance between two vectors.
 *
 *     https://en.wikipedia.org/wiki/Metric_(mathematics)
 *
 * This class file also contains some of the most widely used
 * standard distance evaluation functions that can be accessed
 * via "static" references.
 *
 * All of the default functions perform input validation,
 * and any custom implementation is ought to do so.
 * @see #validateInputVectors(double[], double[])
 */
public interface DistanceMetric {

    /**
     * Validates the two given vectors and measures the distance between them.
     * The last number in both arrays is the ID of a DataRecord label. This
     * number must not be used in calculation. For this reason, "vec1.length - 1"
     * is used whereever possible. Note that since we assume that both given
     * vectors are of equal length, we do not use "vec2.length" in iterations.
     *
     * @param vec1 first vector.
     * @param vec2 second vector.
     *
     * @return the distance between the two specified vectors.
     */
    double measure(double[] vec1, double[] vec2);


    /*

    Below is the implementation of some of the most widely used algorithms.

     */


    /**
     * The distance between two "binary strings" ("bistrings"), or,
     * in other words, between two one-hot encoded vectors. Note that
     * the values in the vectors do not have to be strictly "0" or "1",
     * however, the distance is measured using strict equations ("!="),
     * and is always incremented by one for each feature mismatch.
     *
     * https://en.wikipedia.org/wiki/Hamming_distance
     */
    DistanceMetric HAMMING_DISTANCE = (vec1, vec2) -> {
        validateInputVectors(vec1, vec2);
        double distance = 0;

        for (int feature = 0; feature < vec1.length - 1; feature++)
            if (vec1[feature] != vec2[feature])
                distance += 1.0;

        return distance;
    };


    /**
     * The standard "straight-line" distance between two points in Euclidean space,
     * squared. Not extracting the square root might be essential in scenarios where
     * there is a huge number of distance measurements performed - this way the algorithm
     * will work faster, still keeping the original relative proportions, thus not affecting
     * the output (predictions) of the algorithm in most cases.
     *
     * @see #EUCLIDEAN_DISTANCE
     *
     * https://en.wikipedia.org/wiki/Euclidean_distance
     */
    DistanceMetric EUCLIDEAN_DISTANCE_SQUARE = (vec1, vec2) -> {
        validateInputVectors(vec1, vec2);
        double distance = 0.0;

        for (int feature = 0; feature < vec1.length - 1; feature++) {
            double d = vec1[feature] - vec2[feature];
            distance += d * d;
        }

        return distance;
    };


    /**
     * The standard "straight-line" distance between two points in Euclidean space,
     * as defined by the original formula: d = sqrt( (a1-b1)^2 + ... + (aK-bK)^2 ).
     * Might be comparably slow in scenarios where there is a huge number of distance
     * measurements performed.
     *
     * @see #EUCLIDEAN_DISTANCE_SQUARE
     *
     * https://en.wikipedia.org/wiki/Euclidean_distance
     */
    DistanceMetric EUCLIDEAN_DISTANCE = (vec1, vec2) ->
            Math.sqrt(EUCLIDEAN_DISTANCE_SQUARE.measure(vec1, vec2));


    /**
     * The sum of the absolute differences in the Cartesian coordinates of the vectors.
     * Also known as "taxicab metric", "city block distance", "rectilinear distance",
     * or "snake distance".
     *
     * https://en.wikipedia.org/wiki/Taxicab_geometry
     */
    DistanceMetric MANHATTAN_DISTANCE = (vec1, vec2) -> {
        validateInputVectors(vec1, vec2);

        double distance = 0.0;

        for (int feature = 0; feature < vec1.length - 1; feature++)
            distance += Math.abs(vec1[feature] - vec2[feature]);

        return distance;
    };


    /**
     * Ensures both of the given vectors are safe to be used in distance measurements.
     * Makes direct use of NormalizationFunction#validateInputVector for both vectors.
     *
     * @see NormalizationFunction#validateInputVector(double[])
     *
     * @param vec1 first vector to check.
     * @param vec2 second vector to check.
     *
     * @throws IllegalArgumentException if lengths of the specified vectors do not match,
     *                                  although each of them is valid itself.
     */
    static void validateInputVectors(double[] vec1, double[] vec2) {
        NormalizationFunction.validateInputVector(vec1);
        NormalizationFunction.validateInputVector(vec2);

        if (vec1.length != vec2.length)
            throw new IllegalArgumentException(
                    "vectors must not differ in length: " + vec1.length + "/" + vec2.length);
    }

}

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

import java.util.*;

/**
 * An interface used to initialize LVQ weights.
 *
 * This class file also contains the basic weights initialization
 * functions. The list provided by it should be sufficient for all
 * purposes in most cases.
 *
 * No input validation is performed as it is intended to be done
 * internally by the neural network.
 */
public interface WeightsInitializer {

    /**
     * Initializes all weights vectors in the given set.
     *
     * @param weights target weights vector set.
     * @param trainData all of the samples provided to the neural network in its constructor.
     * @param trainSamples the number of samples to train (<=trainData.length).
     * @param nFeatures the number of features in each sample vector (pattern) plus
     *                  one - the ID of the cluster the sample belongs to (data.length).
     * @param rng the random number generator to use (derived from the neural network).
     */
    void initialize(double[][] weights, double[][] trainData,
                    int trainSamples, int nFeatures, Random rng);


    /*

    Below are the default implementations used in most cases.

     */


    /**
     * Initialize all weights as zeroes. This strategy is usually
     * extremely ineffective for prediction/classification purposes.
     */
    WeightsInitializer ZEROES = (weights, trainData, trainSamples, nFeatures, rng) ->
        {};


    /**
     * Initialize weights with N first records from the given trainData,
     * where N is the configured amount of input to train (trainSamples).
     */
    WeightsInitializer N_FIRST = (weights, trainData, trainSamples, nFeatures, rng) -> {
        for (int sample = 0; sample < trainSamples; sample++)
            System.arraycopy(
                    trainData[sample], 0,
                    weights[sample], 0,
                    nFeatures);
    };


    /**
     * Initialize weights with N random records from the given trainData,
     * where N is the configured amount of input to train (trainSamples).
     *
     * Samples in the weights may repeat an unlimited number of times.
     *
     * @see #N_RANDOM_UNIQUE
     */
    WeightsInitializer N_RANDOM = (weights, trainData, trainSamples, nFeatures, rng) -> {
        for (int sample = 0; sample < trainSamples; sample++) {
            double[] randomTrainSample
                    = trainData[rng.nextInt(trainData.length)];
            System.arraycopy(
                    randomTrainSample, 0,
                    weights[sample], 0,
                    nFeatures);
        }
    };


    /**
     * Initialize weights with N random records from the given trainData,
     * where N is the configured amount of input to train (trainSamples).
     *
     * Train data vectors may not be reused, thus guaranteeing that each
     * weight is initialized to a unique data record from the given train set.
     *
     * This is the standard and the most commonly used strategy.
     *
     * @see #N_RANDOM
     */
    WeightsInitializer N_RANDOM_UNIQUE = (weights, trainData, trainSamples, nFeatures, rng) -> {
        List<double[]> trainDataCopy = new ArrayList<>();
        Collections.addAll(trainDataCopy, trainData);

        for (int sample = 0; sample < trainSamples; sample++) {
            double[] randomTrainSample
                    = trainDataCopy.remove(rng.nextInt(trainDataCopy.size()));
            System.arraycopy(
                    randomTrainSample, 0,
                    weights[sample], 0,
                    nFeatures);
        }
    };


    /**
     * Initialize weights with N random records from the given trainData,
     * where N is the configured amount of input to train (trainSamples),
     * attempting to include an approximately equal number of entries of
     * each label (cluster). This may make the accuracy of predictions of
     * the model more consistent across all labels (clusters), since the
     * amount of data trained from each cluster will vary a lot less.
     *
     * This is especially useful if the numbers of each sample's type
     * in your train data set are very inconsistent (e.g. 50 records of
     * sample type "A", 100 records of type "B", and 500 records of type "C").
     *
     * Train data vectors may not be reused, thus guaranteeing that each
     * weight is initialized to a unique data record from the given train set.
     *
     * @see #N_FIRST_RATIONAL
     * @see #N_RANDOM
     * @see #N_RANDOM_UNIQUE
     */
    WeightsInitializer N_RANDOM_RATIONAL = (weights, trainData, trainSamples, nFeatures, rng) -> {
        List<double[]> trainDataCopy
                = new ArrayList<>(Arrays.asList(trainData));

        if (rng != null) // if rng is not specified then just choose the first N records rationally
            // Randomize the order of entries so that
            // we don't simply read the first N entries.
            Collections.shuffle(trainDataCopy, rng);

        List<double[]> results = new ArrayList<>();
        Set<Integer> addedIndexes = new HashSet<>();

        while (results.size() < trainSamples) {
            Set<Integer> addedLabelIDs = new HashSet<>();

            for (int i = 0; i < trainData.length; i++) {
                if (!addedIndexes.contains(i)) {
                    double[] vec = trainData[i];
                    int labelId = (int) vec[nFeatures - 1]; // last index is not a feature but the label (cluster) ID

                    if (addedLabelIDs.add(labelId)) { // "contains" check + immediate "add" in case of "does not contain"
                        results.add(vec);
                        addedIndexes.add(i);
                    }
                }
            }
        }

        for (int sample = 0; sample < trainSamples; sample++)
            weights[sample] = results.get(sample);
    };


    /**
     * Initialize weights with N first records from the given trainData,
     * where N is the configured amount of input to train (trainSamples),
     * attempting to include an approximately equal number of entries of
     * each label (cluster). This may make the accuracy of predictions of
     * the model more consistent across all labels (clusters), since the
     * amount of data trained from each cluster will vary a lot less.
     *
     * This is especially useful if the numbers of each sample's type
     * in your train data set are very inconsistent (e.g. 50 records of
     * sample type "A", 100 records of type "B", and 500 records of type "C").
     *
     * @see #N_RANDOM_RATIONAL
     * @see #N_FIRST
     */
    WeightsInitializer N_FIRST_RATIONAL = (weights, trainData, trainSamples, nFeatures, rng) ->
            N_RANDOM_RATIONAL.initialize(weights, trainData, trainSamples, nFeatures, null);

}

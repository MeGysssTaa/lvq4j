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

import lombok.NonNull;

/**
 * A skeleton that could be suitable for most types of neural networks.
 */
public interface NeuralNetwork {

    /**
     * Perform input data normalization.
     */
    void normalizeInput();

    /**
     * Initialize neural network weights data.
     */
    void initializeWeights();

    /**
     * Take a snapshot of neural network's current state and save it.
     */
    void saveSnapshot();

    /**
     * Restore the state of the neural network from a previously saved snapshot.
     */
    void restoreFromSnapshot();

    /**
     * Begin the process of training this neural network.
     * The process beings in the current thread.
     */
    void train();

    /**
     * Attempt to predict what label (cluster) would a human
     * have assigned to the specified set of features (pattern).
     *
     * @param testVec the data to classify.
     *
     * @return label ID (cluster index) of the best matching unit
     *         according to the current neural network weights data.
     */
    int classify(@NonNull double[] testVec);

    /**
     * Find what train vector resembles the given test vector the most.
     *
     * @param testVec the data to compare against.
     *
     * @return the train vector that is dimensionally the cloest to the given.
     */
    double[] findBestMatchingUnit(@NonNull double[] testVec);

}

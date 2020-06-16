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
 * An interface used for neural network model persistence.
 */
public interface ModelSerializer {

    /**
     * Save a snapshot of the specified model's current state.
     *
     * @param model model whose current state should be saved.
     *              Guaranteed to be non-null internally.
     */
    void saveSnapshot(NeuralNetwork model);

    /**
     * Load configuration and internal data (including weights and
     * the current learn rate and train iteration (epoch) numbers)
     * into the specified model from a previously saved snapshot.
     *
     * @param model model whose current state should be restored.
     *              Guaranteed to be non-null internally.
     */
    void restoreFromSnapshot(NeuralNetwork model);

}

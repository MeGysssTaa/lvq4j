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

/**
 * This interface enables one with the ability to
 * react a to neural network's state changes.
 */
public interface ModelStateListener {

    /**
     * Called whenever the state of a model is updated. More
     * precisely, this method is called at the end of each training
     * iteration. Performing any slow operations inside this method
     * will slow the training process of the model down significantly as well.
     *
     * @param model the model whose state was just updated.
     * @param currentEpoch the model's current epoch number.
     * @param currentLearnRate the model's current learn rate number.
     * @param currentErrorSquare the model's current total errors square sum number.
     * @param finishedTraining whether this is the last iteration of the model's training
     *                         algorithm (true), or the state is still unfinished, and more
     *                         iterations are expected (false).
     */
    void onStateUpdate(LVQNN model, int currentEpoch, double currentLearnRate,
                       double currentErrorSquare, boolean finishedTraining) throws Throwable;

}

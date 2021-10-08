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

import lombok.Getter;
import lombok.NonNull;

import java.util.List;

/**
 * A handy class that simplifies the work with LVQNN and
 * makes everything relatively more safe, especially for beginners.
 * For making things even easier, it is adviced to use ModelBuilder
 * instead of creating this wrapper manually.
 *
 * @param <T> type of data records to use in training/classification.
 *
 * @see ModelBuilder<T>
 * @see LVQNN for details on implementation.
 */
public class ModelWrapper<T extends DataRecord> {

    /**
     * Raw model. Might be unsafe to use for beginners.
     */
    @Getter
    private final LVQNN model;

    /**
     * Raw list of all input data records.
     */
    @Getter
    private final List<T> inputRecords;

    /**
     * Wraps the specified LVQ model.
     *
     * @param model the model to wrap.
     * @param inputRecords the list of raw input data records.
     *
     * @throws NullPointerException if model or inputRecords is null.
     * @throws IllegalArgumentException if inputRecords.size() != model.getTrainData().length.
     */
    public ModelWrapper(@NonNull LVQNN model, @NonNull List<T> inputRecords) {
        if (inputRecords.size() != model.getTrainData().length)
            throw new IllegalArgumentException("inputRecords list must contain all of the " +
                    "input samples (" + inputRecords.size() + "/" + model.getTrainData().length + ")");

        this.model = model;
        this.inputRecords = inputRecords;
    }

    /**
     * Perform input normalization, initialize weights, and begin training.
     */
    public void preprocessInitializeAndTrain() {
        model.normalizeInput();
        model.initializeWeights();
        model.train();
    }

    /**
     * Restore the model from a snapshot and resume training if needed
     * (if the restored model state is an unfinished state).
     */
    public void restoreFromSnapshotAndResumeTraining() {
        model.restoreFromSnapshot();
        model.train();
    }

}

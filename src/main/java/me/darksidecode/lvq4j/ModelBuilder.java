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

import lombok.NonNull;

import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * A handy class for building and configuring an LVQ model.
 * @param <T> type of data records to use in training/classification.
 *
 * @see ModelWrapper<T>
 * @see LVQNN for details on each configuration option.
 */
public class ModelBuilder<T extends DataRecord> {

    /**
     * The generated model.
     */
    private LVQNN model;

    /**
     * List of all train samples (raw).
     */
    private List<T> allSamples;

    /**
     * Indicates whether the model's learn method decay method has already been set or not.
     */
    private boolean learnDecayMethodSet;

    /**
     * The first method to call when building a model.
     *
     * @param allSamples input records data set.
     * @param trainSamples number of neurons to create weights for.
     */
    public ModelBuilder<T> withTrainData(@NonNull List<T> allSamples, int trainSamples) {
        if (model != null)
            throw new IllegalStateException("model already created");

        // trainSamples to allSamples sizes validation is done internally in the LVQNN constructor
        if (allSamples.isEmpty())
            throw new IllegalArgumentException(
                    "datasets list cannot be empty");

        int totalSamples = allSamples.size();
        double[][] trainData = new double[totalSamples][];
        int firstVecLen = -1; // don't init with first entry here to avoid unnecessary duplicate input validation

        for (int i = 0; i < totalSamples; i++) {
            T record = allSamples.get(i);
            double[] data = record.getData();

            if (Objects.requireNonNull(data, "record data is null: index=" + i).length == 0)
                throw new IllegalArgumentException(
                        "record data is empty: index=" + i);

            if (firstVecLen == -1)
                firstVecLen = data.length;
            else if (data.length != firstVecLen)
                throw new IllegalArgumentException("inconsistent record vecLen: " +
                        "first=" + firstVecLen + ", at index " + i + "=" + data.length);

            trainData[i] = data; // data includes both features set and cluster ID at the very last index
        }

        this.model = new LVQNN(trainData, trainSamples);
        this.allSamples = allSamples;

        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     */
    public ModelBuilder<T> withInputNormalizationFunc(NormalizationFunction inputNormalizationFunc) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        model.setInputNormalizationFunc(inputNormalizationFunc);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     * @throws NullPointerException if weightsInitializer is null.
     */
    public ModelBuilder<T> withWeightsInitializer(@NonNull WeightsInitializer weightsInitializer) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        model.setWeightsInitializer(weightsInitializer);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     * @throws NullPointerException if distanceMetric is null.
     */
    public ModelBuilder<T> withDistanceMetric(@NonNull DistanceMetric distanceMetric) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        model.setDistanceMetric(distanceMetric);
        return this;
    }

    /**
     * Delegates to withRandomNumberGenerator(new Random(rngSeed)).
     * @throws IllegalStateException if withTrainData has not been called yet.
     */
    public ModelBuilder<T> withRandomNumberGenerator(long rngSeed) {
        return withRandomNumberGenerator(new Random(rngSeed));
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     * @throws NullPointerException if rng is null.
     */
    public ModelBuilder<T> withRandomNumberGenerator(@NonNull Random rng) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        model.setRng(rng);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     */
    public ModelBuilder<T> withSerializer(ModelSerializer serializer) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        model.setModelSerializer(serializer);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet, or
     *                               if withSerializer has not been called yet.
     *
     * @throws IllegalArgumentException if snapshotAutoSavePeriod is less than -1.
     */
    public ModelBuilder<T> withSnapshotAutoSavePeriod(int snapshotAutoSavePeriod) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        if (model.getModelSerializer() == null)
            throw new IllegalStateException(
                    "modelSerializer must be set first");

        if (snapshotAutoSavePeriod < -1)
            throw new IllegalArgumentException(
                    "snapshotAutoSavePeriod must be either -1, 0, or a positive integer");

        model.setSnapshotAutoSavePeriod(snapshotAutoSavePeriod);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     * @throws IllegalArgumentException if progressReportPeriod is negative.
     */
    public ModelBuilder<T> withProgressReportPeriod(int progressReportPeriod) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        if (progressReportPeriod < 0)
            throw new IllegalArgumentException(
                    "progressReportPeriod must be either 0 or a positive integer");

        model.setProgressReportPeriod(progressReportPeriod);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     */
    public ModelBuilder<T> withModelStateListener(ModelStateListener modelStateListener) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");
        
        model.setModelStateListener(modelStateListener);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     */
    public ModelBuilder<T> withYieldingTrainThread(boolean yieldMainThread) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        model.setYieldTrainThread(yieldMainThread);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     * @throws IllegalArgumentException if learnRate is not within range (0.0; 1.0].
     */
    public ModelBuilder<T> withLearnRate(double learnRate) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        if (learnRate <= 0.0 || learnRate > 1.0)
            throw new IllegalArgumentException(
                    "learnRate must be in range (0.0; 1.0]");

        model.setLearnRate(learnRate);
        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     * @throws IllegalArgumentException if quitLearnRate is not within range (0.0; 1.0).
     */
    public ModelBuilder<T> withQuitLearnRate(double quitLearnRate) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        if (quitLearnRate <= 0.0 || quitLearnRate >= 1.0)
            throw new IllegalArgumentException(
                    "quitLearnRate must be in range (0.0; 1.0)");

        model.setQuitLearnRate(quitLearnRate);
        return this;
    }

    /**
     * Indicates that the model should use linear learn rate decay.
     *
     * @throws IllegalStateException if withTrainData has not been called yet, or
     *                               if the model's learn rate decay method has already been set.
     */
    public ModelBuilder<T> withLinearLearnRateDecay() {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        if (learnDecayMethodSet)
            throw new IllegalStateException(
                    "learn method decay is already set");

        model.setLinearLearnRateDecay(true);
        learnDecayMethodSet = true;

        return this;
    }

    /**
     * Indicates that the model should use momentum learn rate decay,
     * and use the specified momentum value.
     *
     * @throws IllegalStateException if withTrainData has not been called yet, or
     *                               if the model's learn rate decay method has already been set.
     *
     * @throws IllegalArgumentException if momentum is not within range (0.0; 1.0).
     */
    public ModelBuilder<T> withMomentumLearnRateDecay(double momentum) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        if (learnDecayMethodSet)
            throw new IllegalStateException(
                    "learn method decay is already set");

        if (momentum <= 0.0 || momentum >= 1.0)
            throw new IllegalArgumentException(
                    "momentum must be in range (0.0; 1.0)");

        model.setLinearLearnRateDecay(false);
        model.setMomentum(momentum);
        learnDecayMethodSet = true;

        return this;
    }

    /**
     * @throws IllegalStateException if withTrainData has not been called yet.
     * @throws IllegalArgumentException if maxEpochs is less than 1.
     */
    public ModelBuilder<T> withMaxEpochs(int maxEpochs) {
        if (model == null)
            throw new IllegalStateException("model has not been created yet");

        if (maxEpochs < 1)
            throw new IllegalArgumentException(
                    "maxEpochs must be positive");

        model.setMaxEpochs(maxEpochs);
        return this;
    }

    /**
     * Wraps the configured model and returns the result.
     * @return a ModelWrapper<T> for the configured model.
     */
    public ModelWrapper<T> build() {
        return new ModelWrapper<>(model, allSamples);
    }

}

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

import java.util.Arrays;

/**
 * An appendable implementation of BasicDataRecord.
 * The data array in appendable data records are initialized
 * with NaN values, and must be appended (set) later using
 * special methods. May be comparably unsafe to use.
 *
 * @see BasicDataRecord
 */
public abstract class AppendableDataRecord extends BasicDataRecord {

    /**
     * The default constructor is used in order to restore this record from a String.
     * @see #loadFromString(String)
     */
    public AppendableDataRecord() {}

    /**
     * Constructs a new appendable data record with the given label.
     *
     * Values at indexes 0 through (nFeatures - 1) of the initial data
     * array are filled with NaN values and may be appended (set) later.
     *
     * @param labelText a very brief human-readable label (cluster) description.
     * @param labelId index of the cluster this record belongs to.
     * @param nFeatures number of features in this data record (pattern length).
     *                  The resulting data array length will be equal to (nFeatures + 1).
     *
     * @see LVQNN#trainData for a more detailed documentation of the LVQ4J data vector format.
     */
    @SuppressWarnings ("JavadocReference")
    public AppendableDataRecord(@NonNull String labelText, int labelId, int nFeatures) {
        super(labelText, labelId, nans(nFeatures + 1));
        data[nFeatures] = (double) labelId;
    }

    /**
     * Attempt to append another feature value in this record's data array.
     *
     * @param featureValue value to append.
     *
     * @throws IllegalArgumentException if featureValue is NaN or an Infinite value.
     * @throws IllegalStateException if this record's data array is already filled,
     *                               or, more formally, if (nextIndex() == -1).
     *
     * @return self (for chaining).
     */
    public AppendableDataRecord append(double featureValue) {
        if (Double.isNaN(featureValue) || Double.isInfinite(featureValue))
            throw new IllegalArgumentException(
                    "featureValue cannot be NaN or Infinite");

        int nextIndex = nextIndex();

        if (nextIndex == -1)
            throw new IllegalStateException("already filled");

        data[nextIndex] = featureValue;

        return this;
    }

    /**
     * Attempt to assign data[featureIndex] to the given featureValue.
     *
     * @param featureIndex index of the feature whose value will be overwritten.
     * @param featureValue new value.
     *
     * @throws IllegalArgumentException if featureValue is NaN or an Infinite value, or
     *                                  if featuresIndex is not a valid feature index
     *                                  within this record's data array.
     *
     * @return self (for chaining).
     */
    public AppendableDataRecord set(int featureIndex, double featureValue) {
        if (featureIndex < 0 || featureIndex >= data.length - 1)
            throw new ArrayIndexOutOfBoundsException(
                    "featureIndex is out of bounds: [0; " + (data.length - 1) + ")");

        if (Double.isNaN(featureValue) || Double.isInfinite(featureValue))
            throw new IllegalArgumentException(
                    "featureValue cannot be NaN or Infinite");

        data[featureIndex] = featureValue;

        return this;
    }

    /**
     * Checks whether this data record's features array is already filled.
     *
     * @return true if and only if nextIndex() returns -1, or, in other words,
     *         there are no free features slots left in the data array.
     */
    public boolean isFilled() {
        return nextIndex() == -1;
    }

    /**
     * Attempts to find the first index in the current data array
     * where current value is NaN (not appended (set) yet).
     *
     * @return the first free feature index in the data array, or
     *         -1 if the data array is already filled.
     */
    private int nextIndex() {
        for (int i = 0; i < data.length; i++)
            if (Double.isNaN(data[i]))
                return i;

        return -1;
    }

    /**
     * Creates an array of doubles of the given length and fills it with NaN values.
     *
     * @param len the desired array length (nFeatures + 1).
     * @return an array of doubles of the given length filled with NaN values.
     */
    private static double[] nans(int len) {
        double[] nans = new double[len];
        Arrays.fill(nans, Double.NaN);
        return nans;
    }

}

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
 * An interface for easier data records manipulation.
 *
 * All DataRecord objects must have a default (empty) constructor
 * that can be used to load this DataRecord from a String using
 * method loadFromString(String).
 */
public interface DataRecord {

    /**
     * Returns a human-readable String representation of the label of this DataRecord.
     *
     * For example, for the handwritten letters recognition
     * problem described in class LVQNN this method would return
     * values like "A" for all DataRecords that describe a sample of
     * letter "A", "B" for all DataRecords that describe a sample of
     * letter "B", etc. Label text must not be saved (serialized)
     * among with other data, but rather retrieved using the translate
     * method defined by a particular implementation.
     *
     * Multiple DataRecords may be assigned the same label.
     *
     * @see #getLabelId()
     * @see #labelIdToLabelText(int)
     * @see LVQNN
     *
     * @return the name of the cluster this DataRecord belongs to.
     *         Must be non-null and not empty.
     */
    String getLabelText();

    /**
     * Returns the numerical representation of the label of this DataRecord.
     *
     * For example, for the handwritten letters recognition
     * problem described in class LVQNN this method would return
     * values like 0 for samples of letter "A", 1 for samples of letter "B", etc.
     * Given the label ID of a DataRecord, one must be able to restore its
     * human-readable text using an implementation-dependent translate method.
     *
     * Multiple DataRecords may be assigned the same label.
     *
     * @see #getLabelText()
     * @see #labelIdToLabelText(int)
     * @see #getData()
     * @see LVQNN
     *
     * @return the numerical representation of this label's cluster.
     *         Must be non-negative.
     */
    int getLabelId();

    /**
     * Given the label ID of this DataRecord, translates it into a human-readable
     * non-serialized format. Not used internally. The sole purpose of text labels
     * is to make it easier for a human to understand the output of a neural network.
     *
     * @see #labelTextToLabelId(String)
     * @see #getLabelText()
     * @see #getLabelId()
     *
     * @param labelId the numerical representation of this DataRecord's cluster.
     *
     * @throws IllegalArgumentException if the given labelId is not associated with this
     *                                  particular DataRecord implementation. Guaranteed
     *                                  to be thrown if labelId is negative.
     *
     * @return a human-readable String representation of this DataRecord cluster, such that
     *        labelIdToLabelText(labelTextToLabelId("example")) = "example".
     */
    String labelIdToLabelText(int labelId);

    /**
     * Given a human-readable name of this DataRecord's cluster, translates it into
     * a non-negative index number specific for this particular DataRecord implementation
     * that can be used to identify or classify this DataRecord, or, in other words, for
     * neural network training and prediction purposes.
     *
     * @see #labelIdToLabelText(int)
     * @see #getLabelText()
     * @see #getLabelId()
     *
     * @param labelText the human-readable identifier of this cluster.
     *
     * @throws NullPointerException if labelText is null.
     *
     * @throws IllegalArgumentException if the given labelText is not associated with this
     *                                  particular DataRecord implementation. Guaranteed
     *                                  to be thrown if labelText is empty.
     *
     * @return the index of this DataRecord cluster in the train data set, such that
     *         labelTextToLabelId(labelIdToLabelText(0)) = 0.
     */
    int labelTextToLabelId(@NonNull String labelText);

    /**
     * Returns the numerical representation of this DataRecord that can
     * be fed to the neural network for training or classification
     * needs. The length of this data must be equal for all DataRecords
     * of this type. The very last index of the data array must hold the
     * label ID of this DataRecord.
     *
     * For example, for the handwritten letters recognition
     * problem described in class LVQNN this method would return
     * the array of pixels of the "picture" of a letter plus the ID of the
     * letter at the end. In other words, the vector returned by this method
     * is exhaustive and sufficient for training and prediction needs.
     *
     * @see #getLabelId()
     * @see LVQNN#trainData
     *
     * @return the train pattern describing this DataRecord's sample.
     *         Must be non-null and not empty.
     */
    @SuppressWarnings ("JavadocReference")
    double[] getData();

    /**
     * Save the current data as string, in a format that can be
     * written in a file and then read from it to form this DataRecord
     * back using loadFromString(String). For a human-readable DataRecord
     * representation use toString().
     *
     * @see #loadFromString(String)
     * @see #toString() (implementation-dependent)
     *
     * @throws IllegalStateException if this DataRecord has not been initialized
     *                               or loaded from String using a special method yet.
     *
     * @return saveable, but not necessarily human-readable, representation
     *         of this particular DataRecord and all the data in it as a String.
     */
    String saveToString();

    /**
     * Load data into this DataRecord from a String generated with saveToString().
     *
     * @see #saveToString()
     *
     * @throws IllegalStateException if this DataRecord has already been initialized
     *                               or loaded from String using this method.
     *
     * @throws IllegalArgumentException if the specified String cannot be parsed and/or
     *                                  converted into a DataRecord of this particular type.
     *
     * @param s the string to parse this DataRecord from.
     */
    void loadFromString(@NonNull String s);

}

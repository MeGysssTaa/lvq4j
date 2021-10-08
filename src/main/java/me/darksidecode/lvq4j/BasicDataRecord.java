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
 * The basic, incomplete implementation of a DataRecord. Cannot
 * be used as is due to the lack of some of the required abstract
 * methods, however, may be used as a base for some arbitrary
 * DataRecord implementations.
 *
 * @see DataRecord
 */
public abstract class BasicDataRecord implements DataRecord {

    /**
     * Human-readable label describing the type (cluster) of this record.
     * Should not be stored (serialized). Instead, one should retrieve this
     * using a dedicated method.
     *
     * @see #labelIdToLabelText(int)
     */
    protected String labelText;

    /**
     * Numerical representation of labelText. Stored at the very last index of
     * the data (features) vector.
     *
     * @see #labelText
     * @see #data
     */
    protected int labelId = -1;

    /**
     * The data describing this particular record (sample) - its features
     * set plus the label ID number at the very last index.
     *
     * @see LVQNN#trainData for a more detailed documentation of the LVQ4J data vector format.
     */
    @SuppressWarnings ("JavadocReference")
    protected double[] data;

    /**
     * The default constructor is used in order to restore this record from a String.
     * @see #loadFromString(String)
     */
    public BasicDataRecord() {}

    /**
     * Constructs a new record with the given data.
     *
     * @param labelText a very brief human-readable label (cluster) description.
     * @param labelId index of the cluster this record belongs to.
     * @param data the set of features of this record (its pattern) plus its numerical
     *             labelId at the very last index.
     *
     * @see LVQNN#trainData for a more detailed documentation of the LVQ4J data vector format.
     */
    @SuppressWarnings ("JavadocReference")
    public BasicDataRecord(@NonNull String labelText, int labelId, @NonNull double[] data) {
        if (labelText.isEmpty())
            throw new IllegalArgumentException("labelText cannot be empty");

        if (labelId < 0)
            throw new IllegalArgumentException("labelId cannot be negative");

        if (data.length == 0)
            throw new IllegalArgumentException("data vector cannot be empty");

        this.labelText = labelText;
        this.labelId = labelId;
        this.data = data;
    }

    /**
     * Returns the human-readable name of the cluster this data record belongs to.
     *
     * @see #labelText
     * @see #labelIdToLabelText(int)
     *
     * @return the human-readable name of the cluster this data record belongs to.
     *         The returned String is equal to labelIdToLabelText(labelId).
     */
    @Override
    public String getLabelText() {
        return labelText;
    }

    /**
     * Returns the numerical ID (index) of the cluster this data record belongs to.
     *
     * @see #labelId
     * @see #labelTextToLabelId(String)
     *
     * @return the numerical ID (index) of the cluster this data record belongs to.
     *         The returned int is equal to labelTextToLabelId(labelText).
     */
    @Override
    public int getLabelId() {
        return labelId;
    }

    /**
     * Returns the set of features describing this data record (its pattern)
     * plus its label ID at the very last index of the array.
     *
     * @see LVQNN#trainData for a more detailed documentation of the LVQ4J data vector format.
     *
     * @return the data vector that can be fed to a neural network.
     */
    @SuppressWarnings ("JavadocReference")
    @Override
    public double[] getData() {
        return data;
    }

    /**
     * Serializes this data record in a String. The generated
     * String can then be used to restore this data record.
     *
     * @see #toString() for a human-readable representation of this data record.
     * @see #loadFromString(String)
     *
     * @throws IllegalStateException if any of this record's fields are default.
     *
     * @return a saveable, parsable String representation of this data record.
     */
    @Override
    public String saveToString() {
        if (labelText == null || labelId == -1 || data == null)
            throw new IllegalStateException("not initialized");

        StringBuilder csv = new StringBuilder();

        // Save data (features plus label (cluster) ID)
        for (double d : data)
            csv.append(d).append(',');

        // Delete trailing comma
        csv.deleteCharAt(csv.length() - 1);

        return csv.toString();
    }

    /**
     * Restores this data record from a previously saved saveable, parsable String.
     *
     * @param s the string to parse this DataRecord from.
     *
     * @see #saveToString()
     *
     * @throws NullPointerException if the given string is null.
     * @throws IllegalStateException if any of this record's fields are non-default.
     * @throws IllegalArgumentException if the given string cannot be parsed to form
     *                                  a BasicDataRecord of this particular type.
     */
    @Override
    public void loadFromString(@NonNull String s) {
        if (labelText != null || labelId != -1 || data != null)
            throw new IllegalStateException("already initialized");

        String[] csv = s.split(",");
        int len = csv.length;

        if (len < 2)
            throw new IllegalArgumentException("invalid input CSV string format");

        data = new double[len];

        // Load features
        for (int i = 0; i < len - 1; i++) {
            String featureStr = csv[i];

            try {
                data[i] = Double.parseDouble(featureStr);
            } catch (NumberFormatException ex) {
                throw new IllegalArgumentException(
                        "invalid feature: not a decimal (plaintext): " + featureStr);
            }
        }

        // Load label (cluster) ID
        String labelIdStr = csv[len - 1];

        try {
            double labelIdD = Double.parseDouble(labelIdStr);

            if (labelIdD % 1.0 != 0.0)
                throw new IllegalArgumentException(
                        "invalid labelId: not an integer (not round): " + labelIdStr);

            if (labelIdD < 0)
                throw new IllegalArgumentException(
                        "invalid labelId: cluster index cannot be negative: " + labelIdD);

            data[len - 1] = labelIdD;
            labelId = (int) labelIdD;
        } catch (NumberFormatException ex) {
            try {
                // Try to translate labelText (implementation-dependent) to labelId number.
                labelId = labelTextToLabelId(labelIdStr);
                data[len - 1] = (double) labelId;
            } catch (IllegalArgumentException exc) {
                throw new IllegalArgumentException(
                        "invalid labelId: not a numerical ID and cannot be translated with " +
                                "labelTextToLabelId implemented in " + getClass().getName() + ": " + labelIdStr);
            }
        }

        // Restore cluster name (implementation-dependent)
        labelText = labelIdToLabelText(labelId);
    }

    /**
     * Returns a human-readable representation of this data record.
     * May be useful for debugging purposes. Must not be used for
     * serialization purposes.
     *
     * @see #saveToString() if you are looking for a saveable, parsable String format.
     *
     * @return a human-readable representation of this data record that
     *         contains its labelText, labelId and data.length.
     */
    @Override
    public String toString() {
        if (labelText == null || labelId == -1 || data == null)
            throw new IllegalStateException("not initialized");

        return String.format("%s[labelText=%s, labelId=%d, dataLen=%d]",
                getClass().getSimpleName(), labelText, labelId, data.length);
    }

    /**
     * Checks if this data record is equal to the specified object.
     *
     * @param obj the object to compare against.
     *
     * @throws IllegalStateException if any of this record's fields are default, or
     *                               if both of the below stated conditions (1) and (2)
     *                               are met, but any of the obj's record's fields are default.
     *
     * @return true if and only if all of the following conditions are met:
     *         (1) obj is an intance of BasicDataRecord (and thus is not null), and
     *         (2) obj's implementation class is equal to the one of this particular data record's class, and
     *         (3) obj's label ID (cluster index) is equal to the one of this particular data record, and
     *         (4) obj's data vector length is equal to the one of this particular data record, and finally
     *         (5) obj's data array contents (features) are equal to the ones of this particular data record.
     */
    @Override
    public boolean equals(Object obj) {
        if (labelText == null || labelId == -1 || data == null)
            throw new IllegalStateException("not initialized");

        // Use "getClass() == obj.getClass()" for strict implementation comparison.
        if (obj instanceof BasicDataRecord && getClass() == obj.getClass()) {
            BasicDataRecord other = (BasicDataRecord) obj;

            if (other.labelText == null || other.labelId == -1 || other.data == null)
                throw new IllegalStateException(
                        "the DataRecord to compare against has not been initialized");

            if (labelId == other.labelId && data.length == other.data.length) {
                for (int i = 0; i < data.length; i++)
                    if (data[i] != other.data[i])
                        return false;

                return true;
            } else
                return false;
        } else
            return false;
    }

}

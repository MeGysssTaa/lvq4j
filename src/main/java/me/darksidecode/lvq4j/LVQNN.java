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

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.io.Serializable;
import java.util.Objects;
import java.util.Random;

/**
 * A basic implementation of the LVQ (Learning Vector Quantization)
 * prototype-based supervised classification algorithm:
 *
 *     https://en.wikipedia.org/wiki/Learning_vector_quantization
 *
 * Some methods in this class may not be safe for direct use, especially
 * for beginners, so it is recommended to start with ModelWrapper instead.
 * @see ModelWrapper
 *
 * There is also a handy builder class that makes it easier to setup
 * and configure the model in a comparably safe way. It is adviced to use it.
 * @see ModelBuilder
 */
public class LVQNN implements NeuralNetwork, Serializable {



    /*

    Transient internals. Should not be accessed externally, saved,
    or somehow else associated with any particular LVQNN model.

     */


    /**
     * Used for safe surefire logging, making the
     * logging libraries (dependencies) not required.
     *
     * @see SafeLogger
     */
    private static final transient SafeLogger log = new SafeLogger();

    /**
     * May be used for serialization.
     * @see #modelSerializer
     */
    private static final long serialVersionUID = 8786845050436221160L;



    /*

    Input (train) data.
    Provided directly in the constructor.

     */


    /**
     * Set of sample vectors (records) that this neural network will use for training.
     * Each sample must be of length N and consist of (N - 1) features. At the very
     * last index of each sample, label ID (cluster) of that sample must be put. Multiple
     * records may be assigned the same label ID (cluster).
     *
     * For example, let's suppose we're working on a simple handwritten digit recognition
     * model. Let's suppose our samples consist of 9 "pixels" (numbers). Then, the array
     * length of each sample must be equal to 10, containing the digit itself at the end.
     * In this example, train data may look like this ([] brackets indicate that this value
     * is not a feature of this vector, but its label ID (cluster) - yet it still must be
     * present in the train data):
     *
     *     0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 [0.0]
     *     1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 [0.0]
     *     0.0 1.1 0.0 0.0 1.1 0.0 0.0 1.1 0.0 [1.0]
     *     ...
     *
     * The number of samples to choose from the train data set for
     * actual training is defined by the trainSamples field.
     *
     * @see DataRecord#getData()
     * @see #trainSamples
     */
    @Getter @NonNull
    private final double[][] trainData;

    /**
     * Number of samples to choose from the train data set for weights
     * evaluation. If equal to trainData.length, then all sample vectors
     * from the given train data set will be used for training explicitly.
     * Otherwise, <trainSamples> number of samples will be chosen from the given
     * train data set with strategy defined by the configured weights initializer.
     *
     * @see #trainData
     * @see #weightsInitializer
     */
    @Getter
    private final int trainSamples;



    /*

    Neural network configuration. May be modified before calling
    any of the model's methods (preprocessing/initialization/train).

     */


    /**
     * Input data features normalization algorithm. If null, then
     * the input data will not be normalized, which may result in
     * inconsistent ("unfair") classifications, since the predictions
     * of the model will be biased towards originally larger values
     * due to the nature of most of the distance metrics.
     *
     * A rough example of a case where input normalization would be highly recommended might
     * be a model that tries to predict salary of a person according to one's age, work
     * experience, and the number of family members working in a similar field. In this
     * example, the number of family members will always be very small (e.g. 2.0) compared
     * to the age and experience years numbers (e.g. 45.0 and 15.0). This means that "the
     * number of family members working in a similar field" values will barely make any
     * impact on the model's predictions, ultimately rendering some parts of your data sets
     * useless and resulting in less accurate classifications.
     *
     * @see NormalizationFunction
     *
     * (default = null)
     * (transient: manual serialization required)
     */
    @Getter @Setter
    private transient NormalizationFunction inputNormalizationFunc;

    /**
     * Strategy to use for chosing the <trainSamples> number of samples from the train data set.
     *
     * @see WeightsInitializer
     * @see #trainSamples
     *
     * (default = N_RANDOM)
     * (transient: manual serialization required)
     */
    @Getter @Setter @NonNull
    private transient WeightsInitializer weightsInitializer = WeightsInitializer.N_RANDOM;

    /**
     * Algorithm for measuring the distance between two vectors to use
     * in the BMU (Best Matching Unit) search function, that is, in training
     * and classification.
     *
     * @see #distanceMetric
     * @see #findBestMatchingUnit(double[])
     *
     * (default = EUCLIDEAN_DISTANCE)
     * (transient: manual serialization required)
     */
    @Getter @Setter @NonNull
    private transient DistanceMetric distanceMetric = DistanceMetric.EUCLIDEAN_DISTANCE;

    /**
     * Random number generator used in weights initialization.
     * It is strongly recommended to overwrite this field with
     * an own random number generator of a set seed to ensure
     * strict consistency of testing results.
     *
     * (default = new Random())
     */
    @Getter @Setter @NonNull
    private Random rng = new Random();

    /**
     * Implementation of the serializer of this model. May be used
     * to take snapshots of the neural network's current state, and
     * to restore a neural network's state (and ultimately resume
     * the training process if needed) from a previously saved snapshot.
     * If null, then a NullPointerException will be thrown each time an
     * attempt to perform a (de-)serialization operation will be performed.
     *
     * @see #snapshotAutoSavePeriod
     * @see #saveSnapshot()
     * @see #restoreFromSnapshot()
     *
     * (default = null)
     * (transient: manual serialization required)
     */
    @Getter @Setter
    private transient ModelSerializer modelSerializer;

    /**
     * Defines the behavior of the model's automated snapshots save algorithm:
     *
     *     (a) if -1, then the model will never save its current state automatically;
     *         this is the required configuration if modelSerializer is not set (null);
     *
     *     (b) if 0, then the model will save its current state automatically upon
     *         finishing training, but not during the process itself;
     *
     *     (c) if a positive number N, then the model will automatically save its
     *         current state each Nth epoch. More formally, a snapshot will be automatically
     *         taken whenever (currentEpoch % N) == 0. Setting this field to 1 will make the
     *         model take a snapshot on every single iteration.
     *
     * (default = -1)
     */
    @Getter @Setter
    private int snapshotAutoSavePeriod = -1;

    /**
     * Defines how often should the neural network report its current state
     * during the process of training. More precisely, if this field is set
     * to 0, then the model will never report its current state. Otherwise,
     * if set to a positive number N, the model will report its current state
     * each Nth epoch, or, more formally, whenever (currentEpoch % N) == 0.
     * Setting this field to 1 will make the model report its current state
     * on every single iterations. It is important to note that logging is a
     * type of a very slow operation. Therefore, it is recommended to configure
     * this field to a number that will make the model output its current state
     * only a few times a minute, or ever less frequently if the period of time
     * it takes to finish the training process is considerably long.
     *
     * (default = 5)
     */
    @Getter @Setter
    private int progressReportPeriod = 5;

    /**
     * Listener that will be called whenever the state of this model is updated,
     * that is, at the end of every training algorithm iteration. It is important
     * to note the listener is called synchronously. This means that if it performs
     * some performance-heavy (slow) operations, then it will increase the time it
     * will take for the neural network to finish training significantly. If null,
     * then the model will not expect anything to react to its state changes externally.
     *
     * (default = null)
     * (transient: manual serialization required)
     */
    @Getter @Setter
    private transient ModelStateListener modelStateListener;

    /**
     * Defines whether the training process should yield current thread
     * before each iteration. This may (but not necessarily will) slow
     * the training process, but cause less overall CPU usage.
     *
     * @see Thread#yield()
     *
     * (default = false)
     */
    @Getter @Setter
    private boolean yieldTrainThread;

    /**
     * Defines the initial (base) learn rate of this neural network, i.e. the
     * multiplier used in neurons' weights adjustments during the training process.
     * Note that the model will not assign any new values to this field internally.
     * There is a separate field dedicated for storing the current actual learn rate.
     *
     * @see #currentLearnRate
     * @see #quitLearnRate
     * @see #linearLearnRateDecay
     * @see #momentum
     *
     * (default = 0.3)
     */
    @Getter @Setter
    private double learnRate = 0.3;

    /**
     * Defines the minimum learn rate threshold. If after another training iteration
     * the model's current learn rate drops below this value, the training process
     * is finished no matter the current epoch (iteration) number.
     *
     * @see #currentLearnRate
     * @see #learnRate
     * @see #linearLearnRateDecay
     * @see #momentum
     *
     * (default = 0.001)
     */
    @Getter @Setter
    private double quitLearnRate = 0.001;

    /**
     * Defines the formula that will update model's current learn rate number at the
     * end of every training iteration. If true, then the learn rate number will decay
     * in a linear manner, and the number to subtract from it will be defined automatically
     * according to the current epoch and maxEpochs numbers (example: 0.9, 0.7, 0.5, 0.3, ...).
     * If false, then each iteration the current learn rate number will be multiplied by the
     * momentum field value (example: 0.9, 0.63, 0.441, 0.3087, ...).
     *
     * @see #currentLearnRate
     * @see #learnRate
     * @see #quitLearnRate
     * @see #momentum
     * @see #currentEpoch
     * @see #maxEpochs
     *
     * (default = false)
     */
    @Getter @Setter
    private boolean linearLearnRateDecay;

    /**
     * Defines the number that the current learn rate will be multiplied by after
     * each training algorithm iteration. Only takes affect if linearLearnRateDelay
     * is set to false.
     *
     * @see #currentLearnRate
     * @see #learnRate
     * @see #quitLearnRate
     * @see #linearLearnRateDecay
     *
     * (default = 0.98)
     */
    @Getter @Setter
    private double momentum = 0.98;

    /**
     * Defines the maximum number of training algorithm iterations
     * (also known as "epochs") the neural network is allowed to run.
     *
     * @see #currentEpoch
     *
     * (default = 1000)
     */
    @Getter @Setter
    private int maxEpochs = 1000;



    /*

    Internal (operable) data. Should not be modified externally normally,
    except in a ModelSerializer implementation for saving snapshots or
    restoring a model state from them.

     */



    /**
     * A volatile, thread-safe flag indicating whether the training process
     * should be interrupted forcefully just like it has stopped naturally.
     * May be used from another thread to stop the training process, take a
     * snapshot of model's current state, and resume the process later by
     * constructing a new LVQNN from the saved snapshot.
     *
     * @see #halt()
     * @see #train()
     */
    @Getter
    private volatile boolean halt;

    /**
     * Distance (as per the configured distanceMetric) between the BMU found
     * during the last findBestMatchingUnit method call and the train sample
     * that was dimensionally the closest to that BMU. The default basic
     * implementations of DistanceMetric guarantee that this number is always
     * non-negative.
     *
     * @see DistanceMetric
     * @see #findBestMatchingUnit(double[])
     */
    @Getter @Setter
    private double lastWinnerDistance;

    /**
     * Total errors square sum number achieved during the last training session.
     * Might theoretically be used to measure the accuracy of the model - if across
     * your tests this number is becoming smaller, then you are most-likely doing
     * everything right.
     */
    @Getter @Setter
    private double lastTrainErrorSquare;

    /**
     * Internal output neurons' weights vectors set. Neurons with higher
     * weights are dimensionally closer to the train data records they
     * resemble the most. The weights are adjusted during the training process.
     */
    @Getter @Setter
    private double[][] weights;

    /**
     * Current training algorithm iteration number. Smaller
     * than or equal to the configured maxEpochs number.
     *
     * @see #maxEpochs
     */
    @Getter @Setter
    private int currentEpoch;

    /**
     * Current neural network's learning rate - the number the
     * errors are multiplied during the weights evaluation step.
     * Greater than or equal to the configured quitLearnRate number.
     *
     * @see #quitLearnRate
     */
    @Getter @Setter
    private double currentLearnRate;

    /**
     * Constructs a new Learning Vector Quantization model with the
     * given train data. The input data should not be normalized before
     * being passed into this constructor - instead, one should consider
     * using method LVQNN#normalizeInput() for that. Neural network will
     * not be trained immediately after calling this constructor, neither
     * will the weights be initialied. One should do that all manually
     * for better control.
     *
     * @param trainData train samples dataset, see the respective field of this class.
     * @param trainSamples weights vector size, see the respective field of this class.
     *
     * @see #normalizeInput()
     * @see #initializeWeights()
     * @see #train()
     *
     * @throws NullPointerException if trainData is null.
     * @throws IllegalArgumentException if trainData is empty,
     *                                  if the number of features in each sample (trainData[0].length) is zero, or
     *                                  if trainSamples is not within range [1; trainData.length].
     */
    public LVQNN(@NonNull double[][] trainData, int trainSamples) {
        if (trainData.length == 0)
            throw new IllegalArgumentException("trainData cannot be empty");

        // One must ensure that all train samples contain equal numbers of features externally
        if (trainData[0].length == 0)
            throw new IllegalArgumentException(
                    "trainData features cannot be empty");

        if (trainSamples < 1 || trainSamples > trainData.length)
            throw new IllegalArgumentException("invalid trainSamples number: " +
                    "expected in range [1; " + trainData.length + "], but got " + trainSamples);

        this.trainData = trainData;
        this.trainSamples = trainSamples;
    }

    /**
     * Returns the number of features in each train data sample plus one
     *         (the last index of the features set always stores the ID of
     *         the label the sample is given; see trainData description for
     *         details).
     *
     * @see #trainData
     *
     * @return trainData[0].length
     */
    public int getFeaturesNumber() {
        return trainData[0].length;
    }

    /**
     * Normalize input train data set.
     *
     * This method should not be called more than once on
     * a model, otherwise the behavior of the neural network
     * is undefined.
     *
     * @see #inputNormalizationFunc for details.
     */
    @Override
    public void normalizeInput() {
        if (inputNormalizationFunc != null) {
            long beginTime = System.currentTimeMillis();

            for (int sample = 0; sample < trainSamples; sample++)
                inputNormalizationFunc.normalize(trainData[sample]);

            log.info("Normalized input in {} millis with function {}",
                    System.currentTimeMillis() - beginTime,
                    inputNormalizationFunc.getClass().getName());
        }
    }

    /**
     * Initialize weights. If the model is going to be trained from scratch,
     * this method must be called before starting the training algorithm.
     * Otherwise, if the model is going to be restored from a snapshot, then
     * the weights data must be restored from that snapshot as well, hence
     * dropping the requirement of calling this method.
     *
     * Input normalization must be done before weights initialization,
     * otherwise the behavior of the model is undefined.
     *
     * @see #weightsInitializer for details.
     */
    @Override
    public void initializeWeights() {
        long beginTime = System.currentTimeMillis();
        int nFeatures = getFeaturesNumber();

        weights = new double[trainSamples][nFeatures];
        weightsInitializer.initialize(
                weights, trainData, trainSamples, nFeatures, rng);

        log.info("Initialized weights in {} millis with strategy {}",
                System.currentTimeMillis() - beginTime,
                weightsInitializer.getClass().getName());

        int[] samplesPerCluster = new int[trainSamples];
        log.debug("Samples of each type:");

        for (double[] vec : weights)
            samplesPerCluster[(int) vec[nFeatures - 1]]++;

        for (int i = 0; i < trainSamples; i++) {
            int nRecords = samplesPerCluster[i];
            if (nRecords != 0) log.debug("    {}: {}", i, nRecords);
        }

        log.debug("Not seeing some type(s) here? This means that they " +
                "have not beein included by the configured weights initialization function.");
    }

    /**
     * Save a snapshot of model's current state, including its configuration
     * and all the data marked as "Internal (operable) data" above.
     *
     * @see #modelSerializer
     * @see #snapshotAutoSavePeriod
     *
     * @throws NullPointerException if modelSerializer is null (not configured).
     */
    @Override
    public void saveSnapshot() {
        Objects.requireNonNull(modelSerializer,
                "cannot save model snapshot: no serializer set")
                .saveSnapshot(this);
    }

    /**
     * Restores the model state from a snapshot, including its configuration
     * and all the data marked as "Internal (operable) data" above. If the restored
     * state is unfinished (currentEpoch<maxEpochs and currentLearnRate>quitLearnRate),
     * then invoking method train will resume training from the restored state.
     *
     * @see #modelSerializer
     * @see #train()
     *
     * @throws NullPointerException if modelSerializer is null (not configured).
     */
    @Override
    public void restoreFromSnapshot() {
        Objects.requireNonNull(modelSerializer,
                "cannot restore model from snapshot: no serializer set")
                .restoreFromSnapshot(this);
    }

    /**
     * Copy basic serializable configuration fields from the given LVQNN
     * network into this one. May be used for deserialization purposes.
     *
     * Important: transient configuration fields (e.g. interfaces such as
     * weightsInitializer) are not copied automatically by this method.
     *
     * @param other the model to copy serializable configuration fields from.
     *
     * @throws NullPointerException if other is null.
     * @throws IllegalStateException if this == other (self-copy).
     */
    public void copyBasicConfiguration(@NonNull LVQNN other) {
        if (this == other)
            throw new IllegalStateException("self-copy");

        rng = other.rng;
        snapshotAutoSavePeriod = other.snapshotAutoSavePeriod;
        progressReportPeriod = other.progressReportPeriod;
        yieldTrainThread = other.yieldTrainThread;
        learnRate = other.learnRate;
        quitLearnRate = other.quitLearnRate;
        linearLearnRateDecay = other.linearLearnRateDecay;
        momentum = other.momentum;
        maxEpochs = other.maxEpochs;
    }

    /**
     * Copy internal operable model state data (e.g. weights and current iteration number)
     * from the given LVQNN object into this one. May be used for deserialization purposes.
     * This method assumes that both of the models were provided the same input train data.
     * If not, then further behavior of this particular model is undefined.
     *
     * @param other the model to copy internal neural network state fields from.
     *
     * @throws NullPointerException if other is null.
     * @throws IllegalStateException if this == other (self-copy).
     * @throws IllegalArgumentException if trainData.length != other.trainData.length, or
     *                                  if trainData[0].length != other.trainData[0].length.
     */
    public void copyInternals(@NonNull LVQNN other) {
        if (this == other)
            throw new IllegalStateException("self-copy");

        int nFeatures = other.getFeaturesNumber();

        if (getFeaturesNumber() != nFeatures
                || trainData.length != other.trainData.length)
            throw new IllegalArgumentException("train data vectors do not match");

        lastWinnerDistance = other.lastWinnerDistance;
        lastTrainErrorSquare = other.lastTrainErrorSquare;
        weights = new double[other.weights.length][nFeatures];

        // Don't make the other model share its arrays. Deep-clone them instead.
        for (int i = 0; i < weights.length; i++)
            System.arraycopy(
                    other.weights[i], 0,
                    weights[i], 0,
                    nFeatures);

        currentEpoch = other.currentEpoch;
        currentLearnRate = other.currentLearnRate;
    }

    /**
     * Indicate that the training process should stop forcefully upon its
     * next iteration. Since the halt field is volatile, this method should
     * be considered thread-safe.
     *
     * For details,
     * @see #halt
     * @see #train()
     */
    public void halt() {
        log.info("Thread {} requested the train thread " +
                "to halt upon its next iteration.", Thread.currentThread().getName());

        halt = true;
    }

    /**
     * Begins the neural network training process in the current thread (synchronously).
     *
     * If the model was restored from a previously saved snapshot, and the current state
     * is unfinished (currentEpoch<maxEpochs and currentLearnRate>quitLearnRate), then
     * the model will resume training. Otherwise, it will begin training from scratch.
     *
     * After input and state validation, the method will iteratively run the core LVQ
     * algorithm (https://en.wikipedia.org/wiki/Learning_vector_quantization) while both
     * of the following conditions are met: (1) current iteration (epoch) number is less
     * than the configured maxEpochs number, and (2) current learning rate is greater than
     * the configured quitLearnRate number. For a concrete description of the algorithm and
     * the use of LVQNN configuration see the documentation to the configuration fields and
     * comments in the methods below, including this one.
     *
     * If halt indicator was set to true during the training process, it means that
     * another thread requested the training thread to terminate the training process
     * immediately regardless of the model's current state. In this case, the algorithm
     * will act just like the training has finished naturally (by breaking one or both of
     * the abovestated iteration conditions).
     *
     * It should be taken in consideration that due to the "eager" learning nature of the
     * LVQ algorithm, this method may take a really long time to complete. For this reason,
     * it is adviced to run it asynchronously in a separate thread, and hook a special
     * listener to get notifications about the model's state updates. Note that if an
     * unhandled Exception or an Error would be thrown inside the configured ModelStateListener,
     * the error will simply be logged, and the training will still continue. Since the process
     * of producing Throwables is very slow by its nature, if the attached ModelStateListener
     * throws these unhandled Exceptions/Errors considerably often, then the time it will take
     * for the neural network to finish training will grow significantly.
     *
     * This method should not be ran more than once on a
     * model, otherwise its further behavior is undefined.
     *
     * @see #halt
     * @see #halt()
     *
     * @see #restoreFromSnapshot()
     * @see #normalizeInput()
     * @see #initializeWeights()
     *
     * @see #maxEpochs
     * @see #quitLearnRate
     *
     * @see #modelStateListener
     *
     * @throws IllegalStateException if the weights have not been initialized yet, or
     *                               if the model has been restored from a corrupted snapshot.
     */
    @Override
    public void train() {
        if (weights == null)
            throw new IllegalStateException(
                    "weights must be initialized first");

        if (currentEpoch == 0) {
            if (lastTrainErrorSquare != 0.0)
                throw new IllegalStateException(
                        "corrupted state: epoch=0, lastTrainErrorSquare=" + lastTrainErrorSquare
                                + " - is the snapshot this model was restored from valid?");

            // This model was not loaded from a snapshot. It was created from scratch.
            log.info("Neural network will begin training from scratch.");
            currentLearnRate = learnRate; // begin from the base learn rate
        } else {
            if (lastTrainErrorSquare == 0.0)
                throw new IllegalStateException(
                        "corrupted state: epoch=" + currentEpoch + ", lastTrainErrorSquare=0.0"
                                + " - is the snapshot this model was restored from valid?");

            if (currentEpoch >= maxEpochs || learnRate <= quitLearnRate) {
                // This model was loaded from a snapshot with finished state.
                // In other words, it is already trained. Skip the process.
                log.info("Neural network will not be trained: some of the required " +
                        "conditions are not met. Is the snapshot this model was restored from complete?");

                return;
            }
            
            // This model was loaded from a snapshot with unfinished state. Resume training.
            log.info("Neural network will resume training from epoch {} (error square sum = {})",
                    currentEpoch, lastTrainErrorSquare);
        }

        long beginTime = System.currentTimeMillis();
        int nFeatures = getFeaturesNumber();

        // (1) Start of the LVQ algorithm...
        while (currentEpoch < maxEpochs && currentLearnRate > quitLearnRate) {
            if (halt) {
                log.info("Neural network will stop training forcefully ({})" +
                        " as per external request.", Thread.currentThread().getName());

                // Let some external watcher react to the model's state change if configured.
                callStateUpdateListenerSafely(lastTrainErrorSquare, true);

                break;
            }

            if (yieldTrainThread)
                Thread.yield();

            // ... (1.1) start of an LVQ iteration...
            double sumError = 0.0;

            // ... (1.2) for each sample in the whole set of samples...
            for (double[] sample : trainData) {
                // ... (1.2.1) find the BMU for this sample...
                double[] bmu = findBestMatchingUnit(sample);

                // ... (1.2.2) then for each feature in the data vector (pattern)...
                for (int feature = 0; feature < nFeatures - 1; feature++) {
                    // ... (1.2.2.1) compute the error - the difference between the evaluated BMU's
                    //               feature value and the expected value in the original sample...
                    double error = sample[feature] - bmu[feature];
                    sumError += error * error;

                    // ... (1.2.2.2) if the label (cluster) of the evaluated BMU matches that of
                    //               the original sample, move this neuron closer dimensionally
                    //               (increase its weight), otherwise move it further dimensionally
                    //               (reduce its weight); the "importance" of this particular weight
                    //               adjustment is defined by the current neural network's learn rate...

                    if (bmu[nFeatures - 1] == sample[nFeatures - 1])
                        bmu[feature] += error * currentLearnRate;
                    else
                        bmu[feature] -= error * currentLearnRate;

                    // ... (1.2.2.3) proceed to the next feature...
                }

                // ... (1.2.3) proceed to the next sample...
            }

            // ... (1.3) increment epoch, reduce neural network's learn rate, then repeat
            //           the algorithm if both of the following conditions are met:
            //               (a) epoch < maxEpochs, and
            //               (b) currentLearnRate > quitLearnRate...
            if (linearLearnRateDecay)
                currentLearnRate = learnRate * (1.0 - ((double) currentEpoch / (double) maxEpochs));
            else
                currentLearnRate *= momentum;

            lastTrainErrorSquare = sumError;
            currentEpoch++;

            // ... (1.4) do debugging and take a snapshot of
            //           the model's current state if configured...
            boolean firstOrLastIter = currentEpoch == 0
                    || currentEpoch == maxEpochs
                    || currentLearnRate <= quitLearnRate;

            if (progressReportPeriod > 0 && (firstOrLastIter || currentEpoch % progressReportPeriod == 0))
                log.debug("Finished training epoch {} with " +
                                "learn rate = {}, current error square = {}",
                        currentEpoch, currentLearnRate, sumError);

            if (snapshotAutoSavePeriod > 0 && currentEpoch % snapshotAutoSavePeriod == 0) {
                // Periodical auto-save is enabled.
                saveSnapshot();
                log.debug("Saved an unfinished-state model snapshot automatically.");
            }

            // ... (1.5) let some external watcher react to the
            //           model's state change if configured...
            callStateUpdateListenerSafely(sumError,
                    currentEpoch >= maxEpochs || currentLearnRate <= quitLearnRate);

            // ... (1.6) end of an LVQ iteration, repeat if necessary...
        }

        // ... (2) the resulting weights can now be used to make predictions and classifications.
        //         End of the LVQ algorithm.
        log.info("Training completed. It took {} millis " +
                        "to run {} iterations for a final error square sum of {}",
                System.currentTimeMillis() - beginTime, currentEpoch, lastTrainErrorSquare);

        if (snapshotAutoSavePeriod >= 0) {
            // Auto-save on finish is enabled.
            saveSnapshot();
            log.debug("Saved a finished-state model snapshot automatically.");
        }
    }

    /**
     * If modelStateListener is configured (is not null), then its onStateUpdate
     * method is called safely from the current thread (train thread). If an unhandled
     * Exception/Error would be thrown while executing this method, then it will simply
     * be logged, without interrupting the training process.
     *
     * @see #modelStateListener
     * @see #halt()
     *
     * @param sumError current total errors sum square number.
     * @param lastIteration indicates whether this is the final iteration of the training
     *                      algorithm (either because one or more of the training conditions
     *                      are no longer met after this particular iteration, or another thread
     *                      has made an explicit request to stop training).
     */
    private void callStateUpdateListenerSafely(double sumError, boolean lastIteration) {
        if (modelStateListener != null) {
            try {
                modelStateListener.onStateUpdate(
                        this, currentEpoch, currentLearnRate, sumError, lastIteration);
            } catch (Throwable t) {
                // Simply log this issue and continue training.
                log.error("Unhandled exception/error in model's state " +
                        "update listener " + modelStateListener.getClass().getName(), t);
            }
        }
    }

    /**
     * Attempt to predict what label (cluster) a human would assign to
     * the specified set of features based on the current weights data.
     * More formally, this method will search the BMU for the specified
     * test vector, and return the value at the very last index of its
     * features.
     *
     * @param testVec the data to classify.
     *
     * @see #findBestMatchingUnit(double[])
     *
     * @throws NullPointerException if testVec is null.
     * @throws IllegalArgumentException if testVec is empty.
     * @throws IllegalStateException if the model has not been trained yet.
     *
     * @return label ID (cluster) that this neural network has assigned
     *         to the given test vector at its current state.
     */
    @Override
    public int classify(@NonNull double[] testVec) {
        if (testVec.length == 0)
            throw new IllegalArgumentException("testVec cannot be empty");

        if (lastTrainErrorSquare == 0.0)
            throw new IllegalStateException(
                    "the model has not been trained yet");

        // Attempt to predict what label (cluster) a human would assign to
        // the specified set of features based on the current weights data.
        double[] bmu = findBestMatchingUnit(testVec);
        return (int) bmu[bmu.length - 1];
    }

    /**
     * Finds the BMU (Best Matching Unit) for the specified data vector.
     * The algorithm makes use of the configured distance metric. See
     * the respective field description and comments inside this method
     * for further description.
     *
     * This method is unsafe and normally should not be called externally.
     *
     * @param testVec the data to compare against.
     *
     * @see #distanceMetric
     * @see #classify(double[]) if you are looking for a classification/prediction method.
     *
     * @return a sample vector from the current weights data
     *         that is dimensionally the closest to the given.
     */
    @Override
    public double[] findBestMatchingUnit(@NonNull double[] testVec) {
        double minDist = Double.MAX_VALUE;
        double[] bmu = null;

        // (1) For each neuron in weights...
        for (double[] trainVec : weights) {
            // ... (1.1) measure the distance between this neuron
            //           vector and the specified test data vector...
            double dist = distanceMetric.measure(trainVec, testVec);

            // ... (1.2) if the distance is lower than the distances of all
            //           of the previously tested neurons, then mark this
            //           unit as best matching...
            if (dist < minDist) {
                minDist = dist;
                bmu = trainVec;
            }

            // ... (1.3) check next neuron...
        }

        lastWinnerDistance = minDist;

        // ... (2) return the best matching unit we have found. Since the input
        //         is validated internally before the first call to this method,
        //         it is guaranteed that this method will never return null.
        return bmu;
    }

}

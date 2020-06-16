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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * One of the main goals of LVQ4J is to be an extremely lightweight library.
 * To accomplish this, we are avoiding the direct usage of Slf4j logging, because
 * that would force the user to link extra large libraries. For this reason, we
 * maintain a basic implementation of an own logger in case Slf4j will not be
 * present in the user's runtime (implementation) classpath.
 */
final class SafeLogger {

    /**
     * The logger implementation to use. Either our internal,
     * FallbackLogger, or a org.slf4j.Logger instance - depending
     * on the current classpath.
     */
    private Object impl;

    /**
     * Construct a new SafeLogger instance, automatically checking the current
     * classpath for Slf4j and choosing the appropriate logging implementation.
     */
    SafeLogger() {
        try {
            Class.forName("org.slf4j.Logger");
            impl = LoggerFactory.getLogger("LVQ4J");
        } catch (ClassNotFoundException ex) {
            ((FallbackLogger) (impl = new FallbackLogger())).warn(
                    "Slf4j is not available. LVQ4J will be using its FallbackLogger.");
        }
    }

    /**
     * Send a DEBUG message. Note that all leves are treated the same in our internal FallbackLogger.
     * 
     * @param message the message to log, formatted in the Slf4j manner ({}).
     * @param format format arguments, may be null or empty.
     */
    void debug(@NonNull String message, Object... format) {
        if (impl instanceof FallbackLogger)
            ((FallbackLogger) impl).debug(message, format);
        else
            ((Logger) impl).debug(message, format);
    }

    /**
     * Send an INFO message. Note that all leves are treated the same in our internal FallbackLogger.
     *
     * @param message the message to log, formatted in the Slf4j manner ({}).
     * @param format format arguments, may be null or empty.
     */
    void info (@NonNull String message, Object... format) {
        if (impl instanceof FallbackLogger)
            ((FallbackLogger) impl).info(message, format);
        else
            ((Logger) impl).info(message, format);
    }
    
    /**
     * Send a WARN message. Note that all leves are treated the same in our internal FallbackLogger.
     *
     * @param message the message to log, formatted in the Slf4j manner ({}).
     * @param format format arguments, may be null or empty.
     */
    void warn (@NonNull String message, Object... format) {
        if (impl instanceof FallbackLogger)
            ((FallbackLogger) impl).warn(message, format);
        else
            ((Logger) impl).warn(message, format);
    }

    /**
     * Send an ERROR message. Note that all leves are treated the same in our internal FallbackLogger.
     *
     * @param message the message to log, formatted in the Slf4j manner ({}).
     * @param t the cause of this error whose stacktrace will be
     *          printed right after the given message, cannot be null.
     */
    void error(@NonNull String message, @NonNull Throwable t) {
        if (impl instanceof FallbackLogger)
            ((FallbackLogger) impl).error(message, t);
        else
            ((Logger) impl).error(message, t);
    }



    /*

    A basic, brief, raw FallbackLogger implementation without configuration capabilities.
    DEBUG and INFO levels are printed in the standard output stream (System.out), while
    WARN and ERROR levels are printed in the error stream (System.err). Before being printed,
    the messages are "translated" from the Slf4j format ({}) to the native Java's one (%s).
    Non-formatting percent symbols ("%") in the message are converted into "%%" automatically.

     */



    private final class FallbackLogger {
        private final DateFormat timeFormat = new SimpleDateFormat("HH:mm:ss");

        private void debug(String message, Object... format) {
            log("DEBUG", message, false, format);
        }

        private void info (String message, Object... format) {
            log("INFO ", message, false, format);
        }

        private void warn (String message, Object... format) {
            log("WARN ", message, true, format);
        }

        private void error(String message, Throwable t) {
            log("ERROR", message, true);
            t.printStackTrace();
        }

        private void log(String level, String message, boolean errStream, Object... format) {
            message = message
                    .replace("%", "%%")
                    .replace("{}", "%s");

            String time = timeFormat.format(new Date());
            String formattedMsg = (format == null || format.length == 0)
                    ? message : String.format(message, format);

            (errStream ? System.err : System.out)
                    .printf("[%s] [LVQ4J/SimpleLogger] [%s] %s : %s\n",
                            time, Thread.currentThread().getName(), level, formattedMsg);
        }
    }

}

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
 * Defines how LVQ4J should behave when attempting to log something.
 */
public enum LoggingStrategy {

    /**
     * Attempt to use slf4j/log4j2. If not possible, fallback to the internal simple logger implementation.
     */
    DEFAULT,

    /**
     * Use the internal simple logger implementation, even if slf4j/log4j2 is present and can be used.
     */
    FORCE_INTERNAL,

    /**
     * Do not log anything. Exceptions' stacktraces will still be printed in the standard error stream.
     */
    OFF

}

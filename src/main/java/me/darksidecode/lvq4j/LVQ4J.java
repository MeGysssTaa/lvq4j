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
import lombok.Setter;
import lombok.experimental.UtilityClass;

/**
 * Stores global (static) LVQ4J options. Use with caution.
 */
@UtilityClass
public class LVQ4J {

    /**
     * Defines the strategy LVQ4J will use for logging. This parameter is only used once -- during
     * the creation of the {@link LVQNN} class (during its static initialization), so it only has
     * effect before the first ever us eof {@link LVQNN}.
     * <p>
     * If the value is not overriden with {@link LVQ4J#setLoggingStrategy(LoggingStrategy)}, then
     * LVQ4J will attempt to parse the {@code lvq4j.logging} system property into a {@link LoggingStrategy},
     * or use the default strategy ({@link LoggingStrategy#DEFAULT}) if not possible.
     *
     * @see LoggingStrategy
     */
    @Getter @Setter @NonNull
    private static LoggingStrategy loggingStrategy;

    static {
        try {
            loggingStrategy = LoggingStrategy.valueOf(
                    System.getProperty("lvq4j.logging", "default").toUpperCase());
        } catch (IllegalArgumentException ex) {
            loggingStrategy = LoggingStrategy.DEFAULT;
        }
    }

}

/*
 * Copyright (c) 2022 AtLarge Research
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.opendc.web.runner.deployment;

import io.quarkus.deployment.annotations.BuildProducer;
import io.quarkus.deployment.annotations.BuildStep;
import io.quarkus.deployment.annotations.Record;
import io.quarkus.deployment.builditem.*;
import io.quarkus.runtime.RuntimeValue;
import org.opendc.web.runner.OpenDCRunner;
import org.opendc.web.runner.runtime.OpenDCRunnerRecorder;
import org.opendc.web.runner.runtime.OpenDCRunnerRuntimeConfig;

import java.util.function.BooleanSupplier;

import static io.quarkus.deployment.annotations.ExecutionTime.RUNTIME_INIT;

/**
 * Build processor for the OpenDC web runner Quarkus extension.
 */
public class OpenDCRunnerProcessor {

    private static final String FEATURE = "opendc-runner";

    /**
     * Provide the {@link FeatureBuildItem} for this Quarkus extension.
     */
    @BuildStep(onlyIf = IsIncluded.class)
    public FeatureBuildItem feature() {
        return new FeatureBuildItem(FEATURE);
    }

    /**
     * Build step to index the necessary dependencies for the OpenDC runner.
     */
    @BuildStep
    void addDependencies(BuildProducer<IndexDependencyBuildItem> indexDependency) {
        indexDependency.produce(new IndexDependencyBuildItem("org.opendc.trace", "opendc-trace-opendc"));
        indexDependency.produce(new IndexDependencyBuildItem("org.opendc.trace", "opendc-trace-bitbrains"));
    }

    /**
     * Build step to create the runner service.
     */
    @BuildStep(onlyIf = IsIncluded.class)
    @Record(RUNTIME_INIT)
    ServiceStartBuildItem createRunnerService(OpenDCRunnerRecorder recorder,
                                              OpenDCRunnerRuntimeConfig config,
                                              BuildProducer<OpenDCRunnerBuildItem> runnerBuildItem) {
        RuntimeValue<OpenDCRunner> runner = recorder.createRunner(config);
        runnerBuildItem.produce(new OpenDCRunnerBuildItem(runner));
        return new ServiceStartBuildItem("OpenDCRunnerService");
    }

    /**
     * Build step to start the runner service.
     */
    @BuildStep(onlyIf = IsIncluded.class)
    @Record(RUNTIME_INIT)
    void startRunnerService(ApplicationStartBuildItem start,
                            OpenDCRunnerBuildItem runnerBuildItem,
                            OpenDCRunnerRecorder recorder,
                            OpenDCRunnerRuntimeConfig config,
                            ShutdownContextBuildItem shutdownContextBuildItem) {
        recorder.startRunner(runnerBuildItem.getRunner(), config,shutdownContextBuildItem);
    }

    /**
     * A {@link BooleanSupplier} to determine if the OpenDC web runner extension should be included.
     */
    private static class IsIncluded implements BooleanSupplier {
        OpenDCRunnerConfig config;

        @Override
        public boolean getAsBoolean() {
            return config.include;
        }
    }
}

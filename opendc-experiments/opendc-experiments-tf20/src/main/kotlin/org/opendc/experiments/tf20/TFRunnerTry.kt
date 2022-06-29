package org.opendc.experiments.tf20

import org.opendc.experiments.tf20.core.SimTFDevice
import org.opendc.experiments.tf20.core.TFDevice
import org.opendc.experiments.tf20.distribute.MirroredStrategy
import org.opendc.experiments.tf20.util.MLEnvironmentReader
import org.opendc.simulator.compute.power.LinearPowerModel
import org.opendc.simulator.core.runBlockingSimulation

import java.io.File
//import java.time.Duration
//import java.util.*

import java.util.*

//import kotlin.math.roundToLong

fun main() = TFRunnerTry().main()
//numberOfDCs : Int = 2, workloadSizeGB : Int = 1000, output : String
internal class TFRunnerTry {
    private val experimentSetup = "output/tf20experiments/experimentSetups/setup.csv"
    private val experimentResults = "output/tf20experiments/experimentResults/experiment.csv"

    fun main() {

        val env = "/ibm.json"
        val numberOfDCs = 2
        val batchSize = 128
        val isAlexNet = false
        val workloadSizeGB = 150.0
        val modelSizeGB = 540.0 / 1000
        val learningRounds = 20

        var expCounter = 0

        File("${experimentSetup}").bufferedWriter().use { out ->
            out.write("experimentNumber, env, NumOfDCs, BatchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds\n")
        }

        File("${experimentResults}").bufferedWriter().use { out ->
            out.write("experimentNumber, DCid, NumOfDCs, BatchSize, isAlexNet, DCEnergyUsage, NetworkEnergyUsage, TotalEnergyUse, Time\n")
        }

        //-------------------------------------- Sensitivity Experiments
        // Baseline
        experiment(expCounter, env, numberOfDCs, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds)
        expCounter += 1

        // DC number change
        experiment(expCounter, env, 4, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds)
        expCounter += 1
        experiment(expCounter, env, 8, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds)
        expCounter += 1
        experiment(expCounter, env, 16, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds)
        expCounter += 1

        // Workload size change
        experiment(expCounter, env, numberOfDCs, batchSize/10, isAlexNet, workloadSizeGB/10, modelSizeGB, learningRounds)
        expCounter += 1
        experiment(expCounter, env, numberOfDCs, batchSize/5, isAlexNet, workloadSizeGB/5, modelSizeGB, learningRounds)
        expCounter += 1
        experiment(expCounter, env, numberOfDCs, batchSize*5, isAlexNet, workloadSizeGB*5, modelSizeGB, learningRounds)
        expCounter += 1
        experiment(expCounter, env, numberOfDCs, batchSize*10, isAlexNet, workloadSizeGB*10, modelSizeGB, learningRounds)
        expCounter += 1

        // LearningRounds change
        experiment(expCounter, env, numberOfDCs, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds/10)
        expCounter += 1
        experiment(expCounter, env, numberOfDCs, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds/5)
        expCounter += 1
        experiment(expCounter, env, numberOfDCs, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds*5)
        expCounter += 1
        experiment(expCounter, env, numberOfDCs, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, learningRounds*10)
        expCounter += 1
        //------------------------------------ END Sensitivity Experiments


        //------------------------------------ Realistic Experiments

        //Testing the performance
//        experiment(expCounter, env, numberOfDCs, batchSize/10, isAlexNet, workloadSizeGB/10, modelSizeGB, 5)
//        expCounter += 1
//
//        //Full training
//        experiment(expCounter, env, 6, batchSize, isAlexNet, workloadSizeGB, modelSizeGB, 100)
//        expCounter += 1
//
        //Extreme workload
        experiment(expCounter, env, 2, batchSize*64, isAlexNet, workloadSizeGB*64, modelSizeGB, 5)
        expCounter += 1

        experiment(expCounter, env, 4, batchSize*64, isAlexNet, workloadSizeGB*64, modelSizeGB, 5)
        expCounter += 1

        experiment(expCounter, env, 8, batchSize*64, isAlexNet, workloadSizeGB*64, modelSizeGB, 5)
        expCounter += 1

        //------------------------------------ END Other Experiments


    }

    private fun experiment(experimentNumber: Int, env: String, numberOfDCs: Int, batchSize: Int, isAlexNet: Boolean, workloadSizeGB: Double, modelSizeGB: Double, learningRounds : Int) {
        var content = "${experimentNumber}, ${experimentNumber}, ${env}, ${numberOfDCs}, ${batchSize}, ${isAlexNet}, ${workloadSizeGB}, ${modelSizeGB}, ${learningRounds}\n"
        File("${experimentSetup}").appendText(content)

        simulateDC(experimentNumber, env, batchSize, isAlexNet, numberOfDCs, workloadSizeGB,
            modelSizeGB,"${experimentResults}.csv", false, learningRounds)
        for(i in 1..numberOfDCs)
            simulateDC(experimentNumber, env, batchSize/numberOfDCs, isAlexNet, numberOfDCs, workloadSizeGB,
                modelSizeGB, "${experimentResults}_${experimentNumber}.csv", true, learningRounds)
    }

    private fun simulateDC(experimentNumber: Int, environment: String = "/ibm.json", batchSize: Int, isAlexNet: Boolean,
                           numberOfDCs: Int = 2, workloadSizeGB: Double = 1000.0, modelSizeGB: Double, output: String,
                           isFederated : Boolean, learningRounds: Int) = runBlockingSimulation {
        val envInput = checkNotNull(TFRunnerTry::class.java.getResourceAsStream(environment)) // OR "/ibm.json"
        val defList = MLEnvironmentReader().readEnvironment(envInput)

        val devices = mutableListOf<TFDevice>()
        for (def in defList) {
            val device = SimTFDevice(
                UUID.randomUUID(), def.meta["gpu"] as Boolean, coroutineContext, clock, def.model.cpus[0], def.model.memory[0],
                LinearPowerModel(250.0, 60.0)
            )
            devices.add(device)
        }

        val strategy = MirroredStrategy(devices)
        var model = VGG16(batchSize.toLong())

        if (isAlexNet) {
            model = AlexNet(batchSize.toLong())
        }

        model.use {
            it.compile(strategy)
            it.fit(epochs = 9088 / batchSize, batchSize = batchSize)
        }

        var singleDCEnergySum = 0.0
        var singleDCTimeMillis: Long = 0

        for (device in devices) {
            device.close()
            val stats = device.getDeviceStats()
            singleDCEnergySum += joulesTokWh(stats.energyUsage)
        }
        singleDCTimeMillis = clock.millis()

        val networkUse = calculateNetworkUsage(workloadSizeGB, modelSizeGB, numberOfDCs, isFederated, learningRounds)

        if(!isFederated) {
            var content = "${experimentNumber}, 0, ${numberOfDCs}, ${batchSize}, ${isAlexNet}, ${singleDCEnergySum}, ${networkUse}, ${singleDCEnergySum + networkUse}, ${singleDCTimeMillis}\n"
            File("${experimentResults}").appendText(content)
        } else {
            var content = "${experimentNumber}, 1, ${numberOfDCs}, ${batchSize}, ${isAlexNet}, ${singleDCEnergySum}, ${networkUse}, ${singleDCEnergySum + networkUse}, ${singleDCTimeMillis}\n"
            File("${experimentResults}").appendText(content)
        }
    }

    private fun calculateNetworkUsage(workloadSizeGB: Double, modelSizeGB: Double, numberOfDCs: Int, isFederated: Boolean, learningRounds: Int): Double {
        if(isFederated)
            return 0.023 * (modelSizeGB * (numberOfDCs - 1)) * learningRounds
        return 0.023 * (workloadSizeGB / numberOfDCs) * (numberOfDCs - 1)
    }

    private fun joulesTokWh(transform : Double) : Double {
        return transform / 3600000
    }
}

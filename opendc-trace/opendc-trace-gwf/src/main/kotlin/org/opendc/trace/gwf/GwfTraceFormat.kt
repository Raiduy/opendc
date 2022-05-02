/*
 * Copyright (c) 2021 AtLarge Research
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

package org.opendc.trace.gwf

import com.fasterxml.jackson.dataformat.csv.CsvFactory
import com.fasterxml.jackson.dataformat.csv.CsvParser
import org.opendc.trace.*
import org.opendc.trace.conv.*
import org.opendc.trace.spi.TableDetails
import org.opendc.trace.spi.TraceFormat
import java.nio.file.Path

/**
 * A [TraceFormat] implementation for the GWF trace format.
 */
public class GwfTraceFormat : TraceFormat {
    /**
     * The name of this trace format.
     */
    override val name: String = "gwf"

    /**
     * The [CsvFactory] used to create the parser.
     */
    private val factory = CsvFactory()
        .enable(CsvParser.Feature.ALLOW_COMMENTS)
        .enable(CsvParser.Feature.TRIM_SPACES)

    override fun create(path: Path) {
        throw UnsupportedOperationException("Writing not supported for this format")
    }

    override fun getTables(path: Path): List<String> = listOf(TABLE_TASKS)

    override fun getDetails(path: Path, table: String): TableDetails {
        return when (table) {
            TABLE_TASKS -> TableDetails(
                listOf(
                    TASK_WORKFLOW_ID,
                    TASK_ID,
                    TASK_SUBMIT_TIME,
                    TASK_RUNTIME,
                    TASK_REQ_NCPUS,
                    TASK_ALLOC_NCPUS,
                    TASK_PARENTS,
                ),
                listOf(TASK_WORKFLOW_ID)
            )
            else -> throw IllegalArgumentException("Table $table not supported")
        }
    }

    override fun newReader(path: Path, table: String, projection: List<TableColumn<*>>?): TableReader {
        return when (table) {
            TABLE_TASKS -> GwfTaskTableReader(factory.createParser(path.toFile()))
            else -> throw IllegalArgumentException("Table $table not supported")
        }
    }

    override fun newWriter(path: Path, table: String): TableWriter {
        throw UnsupportedOperationException("Writing not supported for this format")
    }
}

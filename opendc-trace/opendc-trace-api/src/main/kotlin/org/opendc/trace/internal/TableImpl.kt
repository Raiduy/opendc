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

package org.opendc.trace.internal

import org.opendc.trace.Table
import org.opendc.trace.TableColumn
import org.opendc.trace.TableReader
import org.opendc.trace.TableWriter
import java.util.*

/**
 * Internal implementation of [Table].
 */
internal class TableImpl(val trace: TraceImpl, override val name: String) : Table {
    /**
     * The details of this table.
     */
    private val details = trace.format.getDetails(trace.path, name)

    override val columns: List<TableColumn>
        get() = details.columns

    override fun newReader(projection: List<String>?): TableReader {
        return trace.format.newReader(trace.path, name, projection)
    }

    override fun newWriter(): TableWriter = trace.format.newWriter(trace.path, name)

    override fun toString(): String = "Table[name=$name]"

    override fun hashCode(): Int = Objects.hash(trace, name)

    override fun equals(other: Any?): Boolean = other is TableImpl && trace == other.trace && name == other.name
}

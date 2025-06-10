import scala.io.Source
import spatial.dsl._

//takes in CSV, NO COMMA SEPARATION, separate line for each element
//total # of lines = total # of elements

@spatial class MultiheadAttention extends SpatialTest {
    // TODO: tweak the precision
    type T = FixPt[TRUE,_32,_24] // Fixed point notation, [signed, 16 bits, 8 bits fraction]


    override def runtimeArgs = "0"
    def main(args: Array[String]): Unit = {
        val N = 1
        val L_q = 2
        val E = 3

        val input_q_values = loadCSV1D[T]("data_dump/test/input.csv", "\n")
        val weight_q_values = loadCSV1D[T]("data_dump/test/weight.csv", "\n")
        val bias_q_values = loadCSV1D[T]("data_dump/test/bias.csv", "\n")
        val gold_q = loadCSV1D[T]("data_dump/test/output.csv", "\n")

        // DRAMs for inputs
        val input_q_dram = DRAM[T](L_q, N, E)
        val weight_q_dram = DRAM[T](E, E)
        val bias_q_dram = DRAM[T](E)

        // DRAM for output
        val output_q_dram = DRAM[T](L_q, N, E)

        //setMem step 
        setMem(input_q_dram, input_q_values)
        setMem(weight_q_dram, weight_q_values)
        setMem(bias_q_dram, bias_q_values)

        
        Accel {
            val input_q_sram = SRAM[T](L_q, N, E)
            val weight_q_sram = SRAM[T](E, E)
            val bias_q_sram = SRAM[T](E)
            val output_q_sram = SRAM[T](L_q, N, E)

             // Load input matrices into SRAM
            input_q_sram load input_q_dram
            weight_q_sram load weight_q_dram
            bias_q_sram load bias_q_dram

            Foreach(L_q by 1, N by 1, E by 1) { (l, n, e_out) =>
                val sum = Reg[T](0.to[T])
                Foreach(E by 1) { e_in =>
                    sum := sum + input_q_sram(l, n, e_in) * weight_q_sram(e_out, e_in)
                }
                // Divide by 1 or 8 depending on whether we're projecting query or key/value
                output_q_sram(l, n, e_out) = (sum + bias_q_sram(e_out)) / 8
            }

            output_q_dram store output_q_sram
        }

        //test that q_output is created at all; print one specific element, see if any output is created
        val accel_q = getMem(output_q_dram)  // Now q_output_host is a 3D Scala array
        printArray(accel_q, "accel_q")
        printArray(gold_q, "gold_q")

        //TODO: test that the accel_q matches the gold_q
        val cksum = accel_q.zip(gold_q){_==_}.reduce{_&&_}
        assert(cksum)
    }
}

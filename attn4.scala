import scala.io.Source
import spatial.dsl._

//takes in CSV, NO COMMA SEPARATION, separate line for each element
//total # of lines = total # of elements

@spatial class MultiheadAttention4 extends SpatialTest {
    // TODO: tweak the precision
    type T = FixPt[TRUE,_32,_24] // Fixed point notation, [signed, 16 bits, 8 bits fraction]


    override def runtimeArgs = "0"
    def main(args: Array[String]): Unit = {
        val N = 1
        val L_q = 2
        val E = 6

        val attn_output_values = loadCSV1D[T]("data_dump/test4/attn_output.csv", "\n")
        val weight_values = loadCSV1D[T]("data_dump/test4/weight.csv", "\n")
        val bias_values = loadCSV1D[T]("data_dump/test4/bias.csv", "\n")
        val gold_final_output = loadCSV1D[T]("data_dump/test4/final_output.csv", "\n")

        // DRAMs for inputs
        val attn_output_dram = DRAM[T](L_q, N, E)
        val weight_dram = DRAM[T](E, E)
        val bias_dram = DRAM[T](E)

        // DRAM for output
        val output_dram = DRAM[T](L_q, N, E)

        //setMem step 
        setMem(attn_output_dram, attn_output_values)
        setMem(weight_dram, weight_values)
        setMem(bias_dram, bias_values)

        
        Accel {
            val attn_output_sram = SRAM[T](L_q, N, E)
            val weight_sram = SRAM[T](E, E)
            val bias_sram = SRAM[T](E)
            val output_sram = SRAM[T](L_q, N, E)

            // Load input matrices into SRAM
            attn_output_sram load attn_output_dram
            weight_sram load weight_dram
            bias_sram load bias_dram

            Foreach(L_q by 1, N by 1, E by 1) { (l_q, n, e_out) =>
                val sum = Reg[T](0.to[T])
                Foreach(E by 1) { e_in =>
                    sum := sum + attn_output_sram(l_q, n, e_in) * weight_sram(e_out, e_in)
                }
                output_sram(l_q, n, e_out) = sum + bias_sram(e_out)
            }

            output_dram store output_sram
        }

        //test that final_output is created at all; print one specific element, see if any output is created
        val accel_final_output = getMem(output_dram)  // Now q_output_host is a 3D Scala array
        printArray(accel_final_output, "accel_final_output")
        printArray(gold_final_output, "gold_final_output")

        //TODO: test that the accel_final_output matches the gold_final_output
        val cksum = accel_final_output.zip(gold_final_output){_==_}.reduce{_&&_}
        assert(cksum)
    }
}

import scala.io.Source
import spatial.dsl._

//takes in CSV, NO COMMA SEPARATION, separate line for each element
//total # of lines = total # of elements

@spatial class MultiheadAttention3 extends SpatialTest {
    // TODO: tweak the precision
    type T = FixPt[TRUE,_32,_24] // Fixed point notation, [signed, 16 bits, 8 bits fraction]


    override def runtimeArgs = "0"
    def main(args: Array[String]): Unit = {
        val N = 1
        val L_q = 2
        val L_kv = 3
        val E = 6
        val n_heads = 2

        val value_values = loadCSV1D[T]("data_dump/test3/value.csv", "\n")
        val attn_weights_values = loadCSV1D[T]("data_dump/test3/attn_weights.csv", "\n")
        val gold_attn_output = loadCSV1D[T]("data_dump/test3/attn_output.csv", "\n")

        // DRAMs for inputs
        val value_dram = DRAM[T](L_kv, N, E)
        val attn_weights_dram = DRAM[T](N*n_heads, L_q, L_kv)

        // DRAM for output
        val output_dram = DRAM[T](L_q, N, E)

        //setMem step 
        setMem(value_dram, value_values)
        setMem(attn_weights_dram, attn_weights_values)

        
        Accel {
            val value_sram = SRAM[T](L_kv, N, E)
            val attn_weights_sram = SRAM[T](N*n_heads, L_q, L_kv)
            val output_sram = SRAM[T](L_q, N, E)

            // Load input matrices into SRAM
            value_sram load value_dram
            attn_weights_sram load attn_weights_dram

            // Perform batch matrix multiplication
            Foreach(N by 1, n_heads by 1, L_q by 1) { (n, h, l_q) =>
                val head_dim = E / n_heads
                Foreach(head_dim by 1) { d =>
                    val sum = Reg[T](0.to[T])
                    Foreach(L_kv by 1) { l_kv =>
                        sum := sum + attn_weights_sram(n * n_heads + h, l_q, l_kv) * value_sram(l_kv, n, h * head_dim + d)
                    }
                    output_sram(l_q, n, h * head_dim + d) = sum
                }
            }

            output_dram store output_sram
        }

        //test that attn_output is created at all; print one specific element, see if any output is created
        val accel_attn_output = getMem(output_dram)  // Now q_output_host is a 3D Scala array
        printArray(accel_attn_output, "accel_attn_output")
        printArray(gold_attn_output, "gold_attn_output")

        //TODO: test that the accel_attn_output matches the gold_attn_output
        val cksum = accel_attn_output.zip(gold_attn_output){_==_}.reduce{_&&_}
        assert(cksum)
    }
}
